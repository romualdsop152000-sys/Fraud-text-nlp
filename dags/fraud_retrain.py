"""
DAG Airflow : fraud_retrain
Fréquence : hebdomadaire (dimanche à minuit)
Tâches    : check_data → retrain_tfidf → retrain_bert → log_results
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

default_args = {
    "owner": "fraud-nlp",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_data():
    import sys, pandas as pd
    sys.path.insert(0, "/opt/airflow")
    df = pd.read_csv("/opt/airflow/data/processed/train.csv")
    print(f"Train set : {len(df)} exemples.")
    if len(df) < 1000:
        raise ValueError(f"Pas assez de données ({len(df)} < 1000)")


def retrain_tfidf_task(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.pipelines.retrain import retrain_tfidf, promote_tfidf
    model, vec, metrics = retrain_tfidf()
    promote_tfidf(model, vec, metrics)
    context["ti"].xcom_push(key="tfidf_metrics", value=metrics)


def retrain_bert_task(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.pipelines.retrain import retrain_bert, F1_BERT_THRESHOLD
    tfidf_metrics = context["ti"].xcom_pull(task_ids="retrain_tfidf", key="tfidf_metrics")
    f1_tfidf = tfidf_metrics["f1_fraude"]
    if f1_tfidf >= F1_BERT_THRESHOLD:
        raise AirflowSkipException(
            f"F1 TF-IDF={f1_tfidf} >= {F1_BERT_THRESHOLD} — BERT non réentraîné."
        )
    print(f"[BERT] F1 TF-IDF={f1_tfidf} < {F1_BERT_THRESHOLD} — lancement du retraining BERT sur GPU.")
    metrics = retrain_bert()
    context["ti"].xcom_push(key="bert_metrics", value=metrics)


def log_results_task(**context):
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.pipelines.retrain import log_to_mlflow, F1_BERT_THRESHOLD
    tfidf_metrics = context["ti"].xcom_pull(task_ids="retrain_tfidf", key="tfidf_metrics")
    bert_metrics  = context["ti"].xcom_pull(task_ids="retrain_bert",  key="bert_metrics")
    bert_triggered = bert_metrics not in (None, "skipped")
    log_to_mlflow(tfidf_metrics, bert_metrics, True, bert_triggered)


with DAG(
    dag_id="fraud_retrain",
    description="Retraining hebdomadaire TF-IDF + BERT (comparaison)",
    default_args=default_args,
    start_date=datetime(2026, 3, 23),
    schedule_interval="0 0 * * 0",
    catchup=False,
    tags=["fraud", "nlp", "retrain", "mlflow"],
) as dag:

    t1 = PythonOperator(task_id="check_data",    python_callable=check_data)
    t2 = PythonOperator(task_id="retrain_tfidf", python_callable=retrain_tfidf_task)
    t3 = PythonOperator(task_id="retrain_bert",  python_callable=retrain_bert_task,
                        trigger_rule="none_failed")
    t4 = PythonOperator(task_id="log_results",   python_callable=log_results_task,
                        trigger_rule="none_failed")

    t1 >> t2 >> t3 >> t4