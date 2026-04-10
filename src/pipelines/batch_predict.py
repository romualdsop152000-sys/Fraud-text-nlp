"""
Envoie un batch d'emails réels (data/processed/test.csv) à l'API FastAPI.
Utilisé par le DAG Airflow fraud_pipeline.
"""
import random
import requests
import pandas as pd

API_URL  = "http://localhost:8000/predict"
TEST_CSV = "data/processed/test.csv"


def run_batch(api_url: str = API_URL, n: int = 20, seed: int = None) -> int:
    """
    Échantillonne n emails du test set et les envoie à l'API.
    Alterne bert (text_bert) et tfidf (text_tfidf).
    Retourne le nombre de prédictions réussies.
    """
    df = pd.read_csv(TEST_CSV)
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    success = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        model = "bert" if i % 2 == 0 else "tfidf"
        text  = row["text_bert"] if model == "bert" else row["text_tfidf"]

        try:
            r = requests.post(api_url, json={"text": str(text), "model": model}, timeout=15)
            r.raise_for_status()
            data = r.json()
            print(f"[{model}] {data['label']} ({data['probability']:.3f}) — {str(text)[:60]}...")
            success += 1
        except Exception as e:
            print(f"ERREUR [{model}] {str(text)[:40]}... → {e}")

    print(f"\n{success}/{len(sample)} prédictions réussies.")
    return success


if __name__ == "__main__":
    run_batch()