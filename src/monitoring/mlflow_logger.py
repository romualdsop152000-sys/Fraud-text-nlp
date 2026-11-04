import yaml as yml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
from sklearn.metrics import f1_score, average_precision_score, classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader

from src.features.preprocess import clean_text_tfidf, clean_text_bert
from src.models.train_bert import EmailDataset


def load_config(config_path: str = "configs/config.yml") -> dict:
    with open(config_path, "r") as f:
        return yml.safe_load(f)
    
    
def log_tfidf(config_path: str = "configs/config.yml"):
    """Logge le modèle TF - IDF dans MLflow"""
    config = load_config(config_path)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df["text_tfidf"].fillna("")
    y_test = test_df["label"]
    
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    model = joblib.load("models/tfidf_logreg.joblib")
    
    x_vec = vectorizer.transform(X_test)
    y_pred = model.predict(x_vec)
    y_proba = model.predict_proba(x_vec)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    prauc = average_precision_score(y_test, y_proba)
    acc = (y_test == y_pred).mean()
    
    with mlflow.start_run(run_name="TF-IDF + Logistic Regression"):
        
        #Paramètres
        mlflow.log_param("model_type", "TF-IDF + LogisticRegression" +"SMOTE")
        mlflow.log_param("tfidf_max_features", config["model"]["tfidf_max_features"])
        mlflow.log_param("smote_k_neighbors", config["model"]["smote_k_neighbors"])
        mlflow.log_param("ngram_range", (2, 3))
        mlflow.log_param("C", 1.0)
        mlflow.log_param("random_state", config["model"]["random_state"])
        
        # Métriques
        mlflow.log_metric("f1_fraude", round(f1, 4))
        mlflow.log_metric("pr_auc", round(prauc, 4))
        mlflow.log_metric("accuracy", round(acc, 4))
        
        # Artifacts
        mlflow.sklearn.log_model(model, "tfidf_logreg")
        mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")
        
        print(f"TF-IDF logged to MLflow with F1={f1:.4f} | PR-AUC={prauc:.4f} | Acc={acc:.4f}")
        
def log_bert(config_path: str = "configs/config.yml", batch_size: int = None):
    """Logge le modèle DistiBERT dans MLflow"""
    config = load_config(config_path)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    bert_cfg   = config.get("bert", {})
    batch_size = batch_size or bert_cfg.get("batch_size", 16)
    max_length = bert_cfg.get("max_length", 128)
    
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df["text_bert"].fillna("")
    y_test = test_df["label"].tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/bert")
    model = DistilBertForSequenceClassification.from_pretrained("models/bert")
    model.to(device)
    model.eval()
    
    dataset = EmailDataset(X_test, y_test, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    all_preds, all_proba = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_proba.extend(probs)
            
    f1 = f1_score(y_test, all_preds)
    prauc = average_precision_score(y_test, all_proba)
    acc = sum(p == l for p, l in zip(all_preds, y_test))/len(y_test)
    
    with mlflow.start_run(run_name="distilbert-finetune"):
        # Paramètres
        mlflow.log_param("model_type", "DistilBERT fine-tuned")
        mlflow.log_param("base_model", "distilbert-base-uncased")
        mlflow.log_param("epochs", bert_cfg.get("epochs", 3))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("fp16", bert_cfg.get("fp16", True))
        mlflow.log_param("gradient_checkpointing", bert_cfg.get("gradient_checkpointing", True))
        mlflow.log_param("lr", float(bert_cfg.get("lr", 2e-5)))
        mlflow.log_param("warmup", "10%")
        
        # Métriques
        mlflow.log_metric("f1_fraude", round(f1, 4))
        mlflow.log_metric("pr_auc", round(prauc, 4))
        mlflow.log_metric("accuracy", round(acc, 4))
        
        # Artifacts - dossier bert complet
        mlflow.log_artifacts("models/bert", artifact_path = "distilbert_finetune")
        
        print(f"DistilBERT logged to MLflow with F1={f1:.4f} | PR-AUC={prauc:.4f} | Acc={acc:.4f}")
        
        
if __name__ == "__main__":
    print("=== MLflow logging TF-IDF ===")
    log_tfidf()
    print("\n=== MLflow logging BERT ===")
    log_bert()
    print("\nDone. Lance : mlflow ui --port 5000")