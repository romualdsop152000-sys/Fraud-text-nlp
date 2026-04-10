"""
DAG Airflow : fraud_pipeline
Fréquence : toutes les heures
Tâches    : health_check → batch_predict → ingest_to_es
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import requests


default_args = {
    "owner": "fraud-nlp",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def health_check():
    r = requests.get("http://fraud-api:8000/health", timeout=10)
    r.raise_for_status()
    models = r.json().get("models_loaded", [])
    print(f"API OK — modèles chargés : {models}")


def batch_predict():
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.pipelines.batch_predict import run_batch
    count = run_batch(
        api_url="http://fraud-api:8000/predict",
        n=100,
    )
    if count == 0:
        raise ValueError("Aucune prédiction réussie — pipeline arrêté.")
    print(f"{count} prédictions envoyées.")


def ingest_to_es():
    import sys
    sys.path.insert(0, "/opt/airflow")
    from src.monitoring.es_logger import ingest_logs, get_stats, get_client
    count = ingest_logs(log_path="/opt/airflow/logs_app/predictions.jsonl")
    print(f"{count} documents indexés dans Elasticsearch.")
    if count > 0:
        get_stats(get_client())


with DAG(
    dag_id="fraud_pipeline",
    description="Batch predict + indexation ES toutes les heures",
    default_args=default_args,
    start_date=datetime(2026, 3, 23),
    schedule_interval="@hourly",  # "*/5 * * * *", #deja nous commençons par tout les 5 min
    catchup=False,
    tags=["fraud", "nlp", "monitoring"],
) as dag:

    t1 = PythonOperator(task_id="health_check",  python_callable=health_check)
    t2 = PythonOperator(task_id="batch_predict", python_callable=batch_predict)
    t3 = PythonOperator(task_id="ingest_to_es",  python_callable=ingest_to_es)

    t1 >> t2 >> t3