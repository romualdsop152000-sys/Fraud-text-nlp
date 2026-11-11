"""
Streamlit — Interface de démonstration Fraud Detection NLP
Pages :
  1. 🔍 Prédiction live     — envoie un email à la FastAPI
  2. 📊 Métriques modèles   — graphes PR curve, confusion matrix
  3. 📈 MLflow Runs         — historique des runs de retraining
  4. 📋 Historique          — dernières prédictions loggées
"""
import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL      = os.environ.get("API_URL",      "http://fraud-api:8000")
MLFLOW_URL   = os.environ.get("MLFLOW_URI",   "http://mlflow:5000")
AIRFLOW_URL  = os.environ.get("AIRFLOW_URL",  "http://airflow:8080")
LOG_PATH     = os.environ.get("LOG_PATH",     "/app/logs/predictions.jsonl")
IMAGES_DIR   = os.environ.get("IMAGES_DIR",   "/app/docs/images")
# Identifiants Airflow : uniquement depuis l'environnement (.env via compose.yml),
# jamais en dur dans le code. Absents → appels Airflow sans auth (refusés en 401).
_AIRFLOW_USER     = os.environ.get("AIRFLOW_USER")
_AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD")
AIRFLOW_AUTH = (_AIRFLOW_USER, _AIRFLOW_PASSWORD) if _AIRFLOW_USER and _AIRFLOW_PASSWORD else None

st.set_page_config(
    page_title="Fraud Detection NLP",
    page_icon="🛡️",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Fraud Detection NLP")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Prédiction live", "📊 Métriques modèles", "📈 MLflow Runs", "✈️ Airflow", "📋 Historique"],
)

# ── Statut API ─────────────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API ✅ — {len(health.get('models_loaded', []))} modèles chargés")
except Exception:
    st.sidebar.error("API ❌ — non joignable")
    health = {}

st.sidebar.markdown("---")
st.sidebar.caption(f"MLflow : [{MLFLOW_URL}]({MLFLOW_URL})")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Prédiction live
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Prédiction live":
    st.title("🔍 Prédiction live")
    st.markdown("Colle le contenu d'un email — le modèle détermine s'il est frauduleux.")

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area("Contenu de l'email", height=200,
                            placeholder="Dear beneficiary, you have been selected...")
    with col2:
        model = st.selectbox("Modèle", ["tfidf", "bert"])
        st.markdown("---")
        predict_btn = st.button("🚀 Analyser", use_container_width=True)

    if predict_btn:
        if not text.strip():
            st.warning("Veuillez entrer un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    res = requests.post(
                        f"{API_URL}/predict",
                        json={"text": text, "model": model},
                        timeout=30,
                    ).json()

                    label = res.get("label", "?")
                    proba = res.get("probability", 0)
                    pred  = res.get("prediction", -1)

                    if pred == 1:
                        st.error(f"### 🚨 {label}")
                    else:
                        st.success(f"### ✅ {label}")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Prédiction", label)
                    col_b.metric("Probabilité de fraude", f"{proba:.4f}")
                    col_c.metric("Modèle utilisé", res.get("model", model))

                    st.progress(float(proba), text=f"Score de risque : {proba:.1%}")

                except requests.exceptions.ConnectionError:
                    st.error("Impossible de joindre l'API. Vérifiez que le service `fraud-api` est démarré.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    st.markdown("---")
    st.markdown("**Exemple d'email frauduleux :**")
    exemple_fraude = (
        "Dear Beneficiary, I am Mr. James Wilson, the auditor of a reputable bank. "
        "I have discovered that you are the next of kin to a customer who died intestate "
        "and left a deposit of $15.5 million USD. Please respond with your full name "
        "and bank account details to claim this fund."
    )
    if st.button("Charger l'exemple fraude"):
        st.session_state["example"] = exemple_fraude
        st.rerun()

    if "example" in st.session_state:
        st.code(st.session_state["example"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Métriques modèles
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Métriques modèles":
    st.title("📊 Métriques modèles")

    # Résumé chiffres
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TF-IDF — F1",     "0.9407")
    col2.metric("TF-IDF — PR-AUC", "0.9683")
    col3.metric("BERT — F1",       "0.9977")
    col4.metric("BERT — PR-AUC",   "0.9977")

    st.markdown("---")

    # Dernières images générées dans docs/images/
    def latest_images(prefix: str) -> list:
        pattern = os.path.join(IMAGES_DIR, f"{prefix}*.png")
        files = sorted(glob(pattern), reverse=True)
        return files

    tab_tfidf, tab_bert = st.tabs(["TF-IDF + Logistic Regression", "DistilBERT"])

    with tab_tfidf:
        pr_files   = latest_images("tfidf_pr_curve_")
        conf_files = latest_images("tfidf_confusion_")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Precision-Recall Curve")
            if pr_files:
                st.image(pr_files[0], use_container_width=True)
                st.caption(f"Run : {Path(pr_files[0]).stem}")
            else:
                st.info("Aucun graphe disponible — lancer le DAG `fraud_retrain`.")
        with c2:
            st.subheader("Matrice de confusion")
            if conf_files:
                st.image(conf_files[0], use_container_width=True)
                st.caption(f"Run : {Path(conf_files[0]).stem}")
            else:
                st.info("Aucun graphe disponible.")

        st.markdown("---")
        st.markdown("""
**Analyse :**
- **Précision fraude = 99.7%** — seulement 3 faux positifs sur 1 157 emails légitimes
- **Recall fraude = 89.1%** — 94 fraudes manquées sur 864
- **Limite** : modèle sac-de-mots, ne comprend pas la négation ni le contexte
        """)

    with tab_bert:
        curve_files = latest_images("bert_training_curves_")
        pr_files    = latest_images("bert_pr_curve_")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Loss & F1 par epoch")
            if curve_files:
                st.image(curve_files[0], use_container_width=True)
                st.caption(f"Run : {Path(curve_files[0]).stem}")
            else:
                st.info("Aucun graphe disponible — BERT n'a pas encore été entraîné (F1 TF-IDF ≥ 0.92).")
        with c2:
            st.subheader("Precision-Recall Curve")
            if pr_files:
                st.image(pr_files[0], use_container_width=True)
                st.caption(f"Run : {Path(pr_files[0]).stem}")
            else:
                st.info("Aucun graphe disponible.")

        st.markdown("---")
        st.markdown("""
**Analyse :**
- **PR-AUC = 0.9977** — quasi-parfait, ~2 fraudes manquées sur 864
- **Epoch 1 = point optimal** — légère dégradation aux epochs 2-3 (sur-ajustement)
- **Déclenché uniquement si F1 TF-IDF < 0.92** — réentraînement sur GPU RTX 4050
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MLflow Runs
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 MLflow Runs":
    st.title("📈 Historique MLflow")
    st.markdown(f"Interface complète : [{MLFLOW_URL}]({MLFLOW_URL})")

    try:
        # Récupère les expériences
        exps = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            params={"max_results": 10},
            timeout=5,
        ).json().get("experiments", [])

        if not exps:
            st.info("Aucune expérience MLflow trouvée. Lancez le DAG `fraud_retrain`.")
        else:
            exp_names = {e["experiment_id"]: e["name"] for e in exps}
            selected_exp = st.selectbox("Expérience", list(exp_names.values()))
            exp_id = [k for k, v in exp_names.items() if v == selected_exp][0]

            runs_data = requests.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                json={"experiment_ids": [exp_id], "max_results": 20,
                      "order_by": ["start_time DESC"]},
                timeout=5,
            ).json().get("runs", [])

            if not runs_data:
                st.info("Aucun run dans cette expérience.")
            else:
                rows = []
                for run in runs_data:
                    info    = run.get("info", {})
                    metrics = {m["key"]: m["value"]
                               for m in run.get("data", {}).get("metrics", [])}
                    params  = {p["key"]: p["value"]
                               for p in run.get("data", {}).get("params", [])}
                    ts = info.get("start_time", 0)
                    rows.append({
                        "Run"              : info.get("run_name", info.get("run_id", "")[:8]),
                        "Date"             : datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "-",
                        "tfidf_f1"         : round(metrics.get("tfidf_f1_fraude", float("nan")), 4),
                        "tfidf_pr_auc"     : round(metrics.get("tfidf_pr_auc", float("nan")), 4),
                        "bert_f1"          : round(metrics.get("bert_f1_fraude", float("nan")), 4) if "bert_f1_fraude" in metrics else "—",
                        "bert_pr_auc"      : round(metrics.get("bert_pr_auc", float("nan")), 4) if "bert_pr_auc" in metrics else "—",
                        "bert_triggered"   : params.get("bert_triggered", "—"),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Graphe évolution F1 TF-IDF dans le temps
                df_plot = df[df["tfidf_f1"].apply(lambda x: isinstance(x, float))].copy()
                if len(df_plot) > 1:
                    st.subheader("Évolution F1 TF-IDF au fil des runs")
                    st.line_chart(df_plot.set_index("Date")["tfidf_f1"])

    except requests.exceptions.ConnectionError:
        st.warning("MLflow non joignable. Vérifiez que le service `fraud-mlflow` est démarré.")
    except Exception as e:
        st.error(f"Erreur MLflow : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Airflow
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✈️ Airflow":
    st.title("✈️ Airflow — Suivi des DAGs")
    st.markdown(f"Interface complète : [{AIRFLOW_URL}]({AIRFLOW_URL.replace('airflow', 'localhost').replace('8080', '8081')})")

    DAGS = ["fraud_retrain", "fraud_pipeline"]
    STATUS_EMOJI = {
        "success": "✅", "running": "🔄", "failed": "❌",
        "skipped": "⏭️", "queued": "⏳", "upstream_failed": "⚠️",
        "none": "⬜",
    }

    # Descriptions humaines des fréquences
    CRON_LABELS = {
        "0 0 * * 0":   "Hebdomadaire — dimanche à minuit",
        "0 0 * * *":   "Quotidien — à minuit",
        "0 * * * *":   "Toutes les heures",
        "@hourly":     "Toutes les heures",
        "@daily":      "Quotidien",
        "@weekly":     "Hebdomadaire",
        "@monthly":    "Mensuel",
        "*/5 * * * *": "Toutes les 5 minutes",
        "*/15 * * * *":"Toutes les 15 minutes",
        "*/30 * * * *":"Toutes les 30 minutes",
    }

    def describe_schedule(sched) -> str:
        """Retourne une description lisible de la fréquence d'un DAG."""
        if not sched:
            return "Non planifié"
        if isinstance(sched, dict):
            sched_type = sched.get("__type", "")
            value = sched.get("value", "")
            if sched_type == "TimeDelta":
                seconds = int(sched.get("seconds", 0))
                if seconds < 60:
                    return f"Toutes les {seconds}s"
                elif seconds < 3600:
                    return f"Toutes les {seconds // 60} min"
                elif seconds < 86400:
                    return f"Toutes les {seconds // 3600}h"
                else:
                    return f"Tous les {seconds // 86400} jours"
            # CronExpression ou autre
            raw = value or str(sched)
        else:
            raw = str(sched)
        return CRON_LABELS.get(raw.strip(), f"🕐 {raw}")

    # Descriptions métier des DAGs
    DAG_DESCRIPTIONS = {
        "fraud_retrain":  "Réentraîne TF-IDF + BERT (si F1 < 0.92)",
        "fraud_pipeline": "Prédit + indexe les emails dans Elasticsearch",
    }

    try:
        # ── Résumé des DAGs ──────────────────────────────────────────────────
        st.subheader("État des DAGs")
        cols = st.columns(len(DAGS))
        for i, dag_id in enumerate(DAGS):
            resp = requests.get(
                f"{AIRFLOW_URL}/api/v1/dags/{dag_id}",
                auth=AIRFLOW_AUTH, timeout=5,
            )
            if resp.status_code == 200:
                dag = resp.json()
                paused = dag.get("is_paused", True)
                sched_raw = dag.get("schedule_interval")
                freq_label = describe_schedule(sched_raw)
                with cols[i]:
                    st.metric(
                        dag_id,
                        "⏸️ Pausé" if paused else "▶️ Actif",
                    )
                    st.caption(f"**Fréquence :** {freq_label}")
                    st.caption(DAG_DESCRIPTIONS.get(dag_id, ""))
            else:
                cols[i].warning(f"{dag_id} — non joignable ({resp.status_code})")

        st.markdown("---")

        # ── Derniers runs par DAG ────────────────────────────────────────────
        selected_dag = st.selectbox("DAG", DAGS)

        runs_resp = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{selected_dag}/dagRuns",
            params={"limit": 15, "order_by": "-execution_date"},
            auth=AIRFLOW_AUTH, timeout=5,
        )
        runs_resp.raise_for_status()
        dag_runs = runs_resp.json().get("dag_runs", [])

        if not dag_runs:
            st.info(f"Aucun run trouvé pour `{selected_dag}`.")
        else:
            rows = []
            for run in dag_runs:
                state = run.get("state", "none")
                rows.append({
                    "Run ID"     : run.get("dag_run_id", "-"),
                    "État"       : f"{STATUS_EMOJI.get(state, '?')} {state}",
                    "Démarré"    : run.get("start_date", "-")[:19].replace("T", " ") if run.get("start_date") else "-",
                    "Terminé"    : run.get("end_date", "-")[:19].replace("T", " ") if run.get("end_date") else "-",
                    "Type"       : run.get("run_type", "-"),
                })
            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            # ── Tâches du run sélectionné ────────────────────────────────────
            st.markdown("---")
            st.subheader("Détail des tâches")
            selected_run_id = st.selectbox(
                "Run",
                [r["dag_run_id"] for r in dag_runs],
                format_func=lambda x: x,
            )
            tasks_resp = requests.get(
                f"{AIRFLOW_URL}/api/v1/dags/{selected_dag}/dagRuns/{selected_run_id}/taskInstances",
                auth=AIRFLOW_AUTH, timeout=5,
            )
            if tasks_resp.status_code == 200:
                tasks = tasks_resp.json().get("task_instances", [])
                task_rows = []
                for t in tasks:
                    state = t.get("state") or "none"
                    duration = t.get("duration")
                    task_rows.append({
                        "Tâche"     : t.get("task_id", "-"),
                        "État"      : f"{STATUS_EMOJI.get(state, '?')} {state}",
                        "Durée (s)" : round(duration, 1) if duration else "-",
                        "Tentatives": t.get("try_number", 1),
                        "Opérateur" : t.get("operator", "-"),
                    })
                st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)
            else:
                st.warning(f"Impossible de récupérer les tâches ({tasks_resp.status_code}).")

    except requests.exceptions.ConnectionError:
        st.error(f"Airflow non joignable à `{AIRFLOW_URL}`. Vérifiez que le service `airflow` est démarré.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur HTTP Airflow : {e}")
    except Exception as e:
        st.error(f"Erreur : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Historique des prédictions
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Historique":
    st.title("📋 Historique des prédictions")

    if not os.path.exists(LOG_PATH):
        st.info(f"Fichier de logs introuvable : `{LOG_PATH}`")
    else:
        lines = Path(LOG_PATH).read_text(encoding="utf-8").strip().split("\n")
        lines = [l for l in lines if l.strip()]

        if not lines:
            st.info("Aucune prédiction enregistrée.")
        else:
            records = []
            for line in lines[-500:]:  # 500 dernières
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            df = pd.DataFrame(records)

            # Métriques résumées
            total    = len(df)
            fraudes  = int((df["prediction"] == 1).sum()) if "prediction" in df.columns else 0
            legit    = total - fraudes
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total prédictions", total)
            col2.metric("🚨 Fraudes détectées", fraudes)
            col3.metric("✅ Messages légitimes", legit)
            col4.metric("Taux de fraude", f"{fraudes/total:.1%}" if total else "—",
                        delta=f"{legit/total:.1%} légitimes" if total else None,
                        delta_color="normal")

            st.markdown("---")

            # Filtres
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if "model" in df.columns:
                    model_filter = st.multiselect("Modèle", df["model"].unique().tolist(),
                                                  default=df["model"].unique().tolist())
                    df = df[df["model"].isin(model_filter)]
            with f_col2:
                if "label" in df.columns:
                    label_filter = st.multiselect("Label", df["label"].unique().tolist(),
                                                  default=df["label"].unique().tolist())
                    df = df[df["label"].isin(label_filter)]

            # Tableau
            cols_display = [c for c in ["timestamp", "label", "probability", "model", "text_preview"]
                            if c in df.columns]
            st.dataframe(
                df[cols_display].sort_values("timestamp", ascending=False)
                  .head(100)
                  .reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

            # Distribution
            if "label" in df.columns:
                st.markdown("---")
                st.subheader("Distribution des prédictions")
                dist = df["label"].value_counts().reset_index()
                dist.columns = ["Label", "Nombre"]
                st.bar_chart(dist.set_index("Label"))
