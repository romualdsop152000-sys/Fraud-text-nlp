"""
Script de retraining automatique — TF-IDF + BERT conditionnel.
Logique :
  1. Réentraîne TF-IDF (rapide)
  2. Si F1_tfidf < 0.95 → réentraîne BERT sur GPU
  3. Promeut le meilleur modèle
  4. Log dans MLflow
"""
import os
import shutil
import joblib
import pandas as pd
import torch
import yaml as yml
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas besoin d'écran)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

# Résolution absolue depuis le projet, quel que soit le working directory
BASE_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TRAIN_CSV          = os.path.join(BASE_DIR, "data/processed/train.csv")
TEST_CSV           = os.path.join(BASE_DIR, "data/processed/test.csv")
MODEL_PATH         = os.path.join(BASE_DIR, "models/tfidf_logreg.joblib")
VEC_PATH           = os.path.join(BASE_DIR, "models/tfidf_vectorizer.joblib")
BERT_PATH          = os.path.join(BASE_DIR, "models/bert")
CONFIG_PATH        = os.path.join(BASE_DIR, "configs/config.yml")
F1_MIN_IMPROVEMENT = 0.001
F1_BERT_THRESHOLD  = 0.95  # déclenche le retraining BERT si TF-IDF < ce seuil
# Dossier plots : priorité au chemin monté dans le conteneur Airflow,
# sinon fallback sur docs/images/ local
PLOTS_DIR = (
    "/opt/airflow/docs/images"
    if os.path.isdir("/opt/airflow")
    else os.path.join(BASE_DIR, "docs", "images")
)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Fonctions de visualisation ────────────────────────────────────────────────

def _plot_tfidf_results(y_test, y_pred, y_proba, ts: str) -> list:
    """Génère PR curve + matrice de confusion pour TF-IDF."""
    paths = []

    # 1. PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="steelblue")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("TF-IDF — Precision-Recall Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, f"tfidf_pr_curve_{ts}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(path)

    # 2. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Légit", "Fraude"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("TF-IDF — Matrice de confusion")
    path = os.path.join(PLOTS_DIR, f"tfidf_confusion_{ts}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(path)

    return paths


def _plot_bert_results(epoch_losses: list, epoch_f1s: list,
                       labels, probas, ts: str) -> list:
    """Génère loss/epoch + F1/epoch + PR curve pour BERT."""
    paths = []
    epochs = list(range(1, len(epoch_losses) + 1))

    # 1. Loss + F1 par epoch (double axe)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, epoch_losses, "o-", color="tomato",  lw=2, label="Loss")
    ax2.plot(epochs, epoch_f1s,    "s-", color="steelblue", lw=2, label="F1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (train)", color="tomato")
    ax2.set_ylabel("F1 (test)",    color="steelblue")
    ax1.tick_params(axis="y", labelcolor="tomato")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("BERT — Loss & F1 par epoch")
    ax1.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, f"bert_training_curves_{ts}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(path)

    # 2. PR Curve sur l'évaluation finale
    precision, recall, _ = precision_recall_curve(labels, probas)
    pr_auc = average_precision_score(labels, probas)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="mediumseagreen", lw=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="mediumseagreen")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("BERT — Precision-Recall Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, f"bert_pr_curve_{ts}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    paths.append(path)

    return paths


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yml.safe_load(f)


def get_current_f1() -> float:
    """Évalue le modèle TF-IDF actuellement en production."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        return 0.0
    model = joblib.load(MODEL_PATH)
    vec   = joblib.load(VEC_PATH)
    df    = pd.read_csv(TEST_CSV)
    X     = vec.transform(df["text_tfidf"].fillna(""))
    return f1_score(df["label"], model.predict(X))


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def retrain_tfidf() -> tuple:
    """Réentraîne TF-IDF + LogReg. Retourne (model, vectorizer, metrics)."""
    config   = load_config()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    X_train = train_df["text_tfidf"].fillna("")
    y_train = train_df["label"]
    X_test  = test_df["text_tfidf"].fillna("")
    y_test  = test_df["label"]

    vec = TfidfVectorizer(
        max_features=config["model"]["tfidf_max_features"],
        ngram_range=(2, 3),
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    sm = SMOTE(k_neighbors=config["model"]["smote_k_neighbors"],
               random_state=config["model"]["random_state"])
    X_res, y_res = sm.fit_resample(X_tr, y_train)

    model = LogisticRegression(max_iter=1000, C=1.0,
                               random_state=config["model"]["random_state"])
    model.fit(X_res, y_res)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "f1_fraude": round(f1_score(y_test, y_pred), 4),
        "pr_auc":    round(average_precision_score(y_test, y_proba), 4),
    }
    print(f"[TF-IDF] F1={metrics['f1_fraude']} | PR-AUC={metrics['pr_auc']}")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["plot_paths"] = _plot_tfidf_results(y_test.tolist(), y_pred.tolist(), y_proba.tolist(), ts)
    print(f"[TF-IDF] Graphes sauvegardés : {metrics['plot_paths']}")
    return model, vec, metrics


def promote_tfidf(model, vec, metrics: dict) -> bool:
    """Promeut TF-IDF si meilleur que le modèle en prod."""
    current_f1 = get_current_f1()
    new_f1     = metrics["f1_fraude"]
    print(f"[TF-IDF] F1 prod={current_f1:.4f} | F1 nouveau={new_f1:.4f}")

    if new_f1 >= current_f1 + F1_MIN_IMPROVEMENT or current_f1 == 0.0:
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, MODEL_PATH.replace(".joblib", "_backup.joblib"))
            shutil.copy(VEC_PATH,   VEC_PATH.replace(".joblib",   "_backup.joblib"))
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vec,   VEC_PATH)
        print(f"[TF-IDF]  Modèle promu (F1={new_f1:.4f})")
        return True
    print(f"[TF-IDF] ⏭ Non promu — pas d'amélioration")
    return False


# ── BERT (conditionnel) ───────────────────────────────────────────────────────

def retrain_bert(epochs: int = None, batch_size: int = None, max_length: int = None) -> dict:
    """
    Réentraîne DistilBERT sur GPU.
    Appelé uniquement si F1_tfidf < F1_BERT_THRESHOLD.
    Paramètres GPU optimisés pour RTX 4050 Laptop (6 Go VRAM).
    """
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    from transformers import (DistilBertTokenizerFast,
                              DistilBertForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from torch.optim import AdamW

    config     = load_config()
    bert_cfg   = config.get("bert", {})
    epochs     = epochs     or bert_cfg.get("epochs",     3)
    batch_size = batch_size or bert_cfg.get("batch_size", 8)
    max_length = max_length or bert_cfg.get("max_length", 128)
    use_fp16   = bert_cfg.get("fp16", True)
    use_grad_ckpt = bert_cfg.get("gradient_checkpointing", True)
    lr         = float(bert_cfg.get("retrain_lr", 5e-6))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = use_fp16 and device.type == "cuda"  # fp16 inutile sur CPU
    print(f"[BERT] Device: {device} | fp16: {use_fp16} | grad_ckpt: {use_grad_ckpt}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    X_train, y_train = train_df["text_bert"].fillna("").tolist(), train_df["label"].tolist()
    X_test,  y_test  = test_df["text_bert"].fillna("").tolist(),  test_df["label"].tolist()

    # Tokenisation lazy (par échantillon) → évite de charger tout le corpus en RAM
    class EmailDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts     = texts
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_len   = max_len
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            enc = self.tokenizer(
                self.texts[i], truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt"
            )
            return {"input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels":         torch.tensor(self.labels[i], dtype=torch.long)}

    tokenizer    = DistilBertTokenizerFast.from_pretrained(BERT_PATH)
    train_loader = DataLoader(EmailDataset(X_train, y_train, tokenizer, max_length),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(EmailDataset(X_test, y_test, tokenizer, max_length),
                              batch_size=batch_size,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_PATH, num_labels=2)
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()  # économise ~30% VRAM
    model.to(device)

    optimizer   = AdamW(model.parameters(), lr=lr)  # bert.retrain_lr dans config.yml
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)
    scaler = GradScaler(enabled=use_fp16)

    epoch_losses, epoch_f1s = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast(enabled=use_fp16):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += out.loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(round(avg_loss, 4))

        # Évaluation rapide à la fin de chaque epoch pour le graphe
        model.eval()
        ep_preds, ep_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                with autocast(enabled=use_fp16):
                    out = model(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device))
                ep_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
                ep_labels.extend(batch["labels"].numpy())
        ep_f1 = round(f1_score(ep_labels, ep_preds), 4)
        epoch_f1s.append(ep_f1)
        print(f"[BERT] Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} | F1: {ep_f1:.4f}")

    # Évaluation
    model.eval()
    preds, labels, probas = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            with autocast(enabled=use_fp16):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
            p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probas.extend(p)
            preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())

    if device.type == "cuda":
        torch.cuda.empty_cache()  # libère la VRAM après l'entraînement

    metrics = {
        "f1_fraude": round(f1_score(labels, preds), 4),
        "pr_auc":    round(average_precision_score(labels, probas), 4),
    }
    print(f"[BERT] F1={metrics['f1_fraude']} | PR-AUC={metrics['pr_auc']}")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["plot_paths"] = _plot_bert_results(
        epoch_losses, epoch_f1s, labels, [float(p) for p in probas], ts)
    print(f"[BERT] Graphes sauvegardés : {metrics['plot_paths']}")

    # Sauvegarde
    os.makedirs(BERT_PATH, exist_ok=True)
    model.save_pretrained(BERT_PATH)
    tokenizer.save_pretrained(BERT_PATH)
    print(f"[BERT]  Modèle sauvegardé dans {BERT_PATH}")
    return metrics


# ── Pipeline principal ────────────────────────────────────────────────────────

def log_to_mlflow(tfidf_metrics: dict, bert_metrics: dict | None,
                  tfidf_promoted: bool, bert_triggered: bool) -> None:
    import os as _os
    # MLflow 3.x bloque les Host headers non-localhost (DNS rebinding protection)
    # On le désactive côté client avant tout appel
    _os.environ["MLFLOW_DISABLE_HOST_CHECK"] = "true"
    tracking_uri = _os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection-retrain")
    with mlflow.start_run(run_name="retrain-pipeline"):
        mlflow.log_params({
            "bert_triggered":    bert_triggered,
            "tfidf_promoted":    tfidf_promoted,
            "f1_bert_threshold": F1_BERT_THRESHOLD,
        })
        mlflow.log_metrics({f"tfidf_{k}": v for k, v in tfidf_metrics.items()
                             if isinstance(v, (int, float))})
        if bert_metrics and bert_metrics != "skipped":
            mlflow.log_metrics({f"bert_{k}": v for k, v in bert_metrics.items()
                                 if isinstance(v, (int, float))})

        # Log des graphes comme artifacts
        for path in tfidf_metrics.get("plot_paths", []):
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="plots/tfidf")
        if bert_metrics and bert_metrics != "skipped":
            for path in bert_metrics.get("plot_paths", []):
                if os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path="plots/bert")
    print("[MLflow] Run enregistré.")


def run_retrain_pipeline() -> None:
    print("=== Retraining TF-IDF ===")
    model, vec, tfidf_metrics = retrain_tfidf()
    tfidf_promoted = promote_tfidf(model, vec, tfidf_metrics)

    f1_tfidf      = tfidf_metrics["f1_fraude"]
    bert_triggered = f1_tfidf < F1_BERT_THRESHOLD
    bert_metrics   = None

    if bert_triggered:
        print(f"\n=== F1 TF-IDF={f1_tfidf} < {F1_BERT_THRESHOLD} → Retraining BERT (GPU) ===")
        bert_metrics = retrain_bert()
    else:
        print(f"\n=== F1 TF-IDF={f1_tfidf} >= {F1_BERT_THRESHOLD} → BERT non réentraîné ===")

    log_to_mlflow(tfidf_metrics, bert_metrics, tfidf_promoted, bert_triggered)
    print("\n=== Pipeline terminé ===")

if __name__ == "__main__":
    run_retrain_pipeline()