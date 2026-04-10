import os
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal

import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.features.preprocess import clean_text_tfidf, clean_text_bert


# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

PREDICTIONS_LOG = "logs/predictions.jsonl"


# ── Chargement des modèles au démarrage ────────────────────────────────────────
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les deux modèles une seule fois au démarrage, libère à l'arrêt."""
    logger.info("Chargement des modèles...")

    # TF-IDF
    models["tfidf_vectorizer"] = joblib.load("models/tfidf_vectorizer.joblib")
    models["tfidf_logreg"]     = joblib.load("models/tfidf_logreg.joblib")
    logger.info("TF-IDF chargé")

    # DistilBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["bert_device"]    = device
    models["bert_tokenizer"] = DistilBertTokenizerFast.from_pretrained("models/bert")
    bert_model = DistilBertForSequenceClassification.from_pretrained("models/bert")
    bert_model.to(device)
    bert_model.eval()
    models["bert_model"] = bert_model
    logger.info(f"DistilBERT chargé — device : {device}")

    yield  # L'application tourne ici

    models.clear()
    logger.info("Modèles libérés")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Email Detection API",
    description="Détecte les emails frauduleux via TF-IDF ou DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas Pydantic ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: Literal["tfidf", "bert"] = "bert"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide.")
        return v


class PredictResponse(BaseModel):
    prediction: int    # 0 = Non-Fraude, 1 = Fraude
    label: str         # "Non-Fraude" ou "Fraude"
    probability: float # Probabilité de fraude (classe 1)
    model: str         # Modèle utilisé pour cette prédiction


# ── Inférence ──────────────────────────────────────────────────────────────────
def predict_tfidf(text: str) -> tuple[int, float]:
    """Nettoyage TF-IDF → vectorisation → prédiction Logistic Regression."""
    cleaned = clean_text_tfidf(text)
    vec     = models["tfidf_vectorizer"].transform([cleaned])
    proba   = models["tfidf_logreg"].predict_proba(vec)[0][1]
    pred    = int(proba >= 0.5)
    return pred, round(float(proba), 4)


def predict_bert(text: str) -> tuple[int, float]:
    """Nettoyage minimal → tokenisation → inférence DistilBERT."""
    cleaned   = clean_text_bert(text)
    tokenizer = models["bert_tokenizer"]
    model     = models["bert_model"]
    device    = models["bert_device"]

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        proba  = torch.softmax(logits, dim=1)[0][1].item()

    pred = int(proba >= 0.5)
    return pred, round(proba, 4)


def log_prediction(text: str, model_name: str, prediction: int, probability: float) -> None:
    """Journalise chaque prédiction dans logs/predictions.jsonl (une ligne JSON par requête)."""
    entry = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "model":        model_name,
        "text_preview": text[:150],   # Aperçu tronqué — pas de données sensibles complètes
        "prediction":   prediction,
        "label":        "Fraude" if prediction == 1 else "Non-Fraude",
        "probability":  probability,
    }
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Vérifie que l'API est active et que les modèles sont chargés."""
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Prédit si un email est frauduleux.

    - **text** : contenu brut de l'email
    - **model** : `"bert"` (défaut) ou `"tfidf"`
    """
    try:
        if request.model == "tfidf":
            pred, proba = predict_tfidf(request.text)
        else:
            pred, proba = predict_bert(request.text)
    except Exception as e:
        logger.error(f"Erreur prédiction — model={request.model} : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

    log_prediction(request.text, request.model, pred, proba)
    logger.info(f"model={request.model} prediction={pred} proba={proba}")

    return PredictResponse(
        prediction=pred,
        label="Fraude" if pred == 1 else "Non-Fraude",
        probability=proba,
        model=request.model,
    )
