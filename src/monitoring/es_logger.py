import json
import os
import time
from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers

INDEX_NAME = "fraud-predictions"
PREDICTIONS_LOG = "logs/predictions.jsonl"

def get_client() -> Elasticsearch:
    """Retourne un client ES — utilise ES_HOST si défini (Docker), sinon localhost."""
    host = os.getenv("ES_HOST", "localhost")
    return Elasticsearch(f"http://{host}:9200")


def create_index(es:Elasticsearch) -> None:
    """Cree l'index avec le mapping approprié s'il n'existe pas"""
    if es.indices.exists(index=INDEX_NAME):
        return
    
    mapping = {
        "mappings":{
            "properties":{
                "timestamp":  {"type": "date"},
                "model":      {"type": "keyword"},
                "prediction": {"type": "integer"},
                "label":      {"type": "keyword"},
                "probability":{"type": "float"},
                "text_preview":{"type": "keyword"},
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Index '{INDEX_NAME}' crée.")
    
def ingest_logs(log_path: str = PREDICTIONS_LOG) -> int:
    """
    Lit logs/predictions.jsonl et ingère les entrées dans Elasticsearch.
    Retourne le nombre de documents indexés.
    """
    es = get_client()
    
    # Attendre qu'Elasticsearch soit prêt
    for attempt in range(10):
        try:
            if es.ping():
                break
        except Exception:
            pass
        print(f"Attente Elasticsearch... (tentative {attempt + 1}/10)")
        time.sleep(3)
    else:
        raise RuntimeError("Elasticsearch non disponible après 30 secondes.")
    
    create_index(es)
    
    # Lire le fichier JSONL
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print("Aucune prédiction à ingerer.")
            return 0
        
        # Bulk indexer les documents
        actions = [
            {
                "_index": INDEX_NAME,
                "_source": json.loads(line),
            }
            for line in lines
        ]
        
        success, errors = helpers.bulk(es, actions, raise_on_error=False)
        print(f"{success} documents indexés dans '{INDEX_NAME}'.")
        if errors:
            print(f"Erreurs rencontrées lors de l'indexation : {errors}")
            
        return success
    
    
def get_stats(es: Elasticsearch) -> None:
    """Affiche des statistiques de base sur les prédictions indexées."""
    res = es.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "par_label":{
                    "terms": {"field": "label"}
                },
                "par_model":{
                    "terms": {"field": "model"}
                },
                "proba_moy": {
                    "avg": {"field": "probability"}
                }
            }
        }
    )
    aggs = res["aggregations"]
    print("\n===Stats Elasticsearch===")
    print("Distribution par label:")
    for bucket in aggs["par_label"]["buckets"]:
        print(f"  {bucket['key']:15s}: {bucket['doc_count']}")
    print("Distribution par modèle:")
    for bucket in aggs["par_model"]["buckets"]:
        print(f"  {bucket['key']:10s}: {bucket['doc_count']}")
    proba = aggs['proba_moy']['value']
    print(f"Probabilité moyenne de fraude: {proba:.4f}" if proba is not None else "Probabilité moyenne : N/A (index en cours de refresh)")
    
    
if __name__ == "__main__":
    count = ingest_logs()
    if count > 0:
        es = get_client()
        get_stats(es)
    print("\nDone. Kibana : http://localhost:5601")
    