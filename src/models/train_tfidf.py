import yaml as yml
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE

def load_config(config_path: str = "configs/config.yml") -> dict:
    with open(config_path, "r") as f:
        return yml.safe_load(f)
    
def train_tfidf(config_path: str = "configs/config.yml"):
    config = load_config(config_path)
    
    # Chargement des données prétraitées
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df["text_tfidf"].fillna("")
    y_train = train_df["label"]
    X_test = test_df["text_tfidf"].fillna("")
    y_test = test_df["label"]
    
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"Distribution avant SMOTE -> {y_train.value_counts().to_dict()}")
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=config["model"]["tfidf_max_features"],
        ngram_range=(2, 3)  # bigrammes et trigrammes
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # SMOTE pour équilibrer les classes
    sm = SMOTE(
        k_neighbors=config["model"]["smote_k_neighbors"],
        random_state=config["model"]["random_state"]
    )
    X_resampled, y_resampled = sm.fit_resample(X_train_tfidf, y_train)
    print(f"Distribution après SMOTE -> {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Entraînement du modèle de régression logistique
    model = LogisticRegression(
        max_iter=1000,
        random_state=config["model"]["random_state"],
        C=1.0
    )
    model.fit(X_resampled, y_resampled)
    
    # Évaluation
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    print("\n=== Résultats TF-IDF + Logistic Regression + SMOTE ===")
    print(classification_report(y_test, y_pred, target_names=["Non-Fraude", "Fraude"]))
    print(f"F1 (fraude)  : {f1_score(y_test, y_pred):.4f}")
    print(f"PR-AUC       : {average_precision_score(y_test, y_proba):.4f}")
    
    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,      "models/tfidf_logreg.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    print("\nModèle sauvegardé : models/tfidf_logreg.joblib")
    print("Vectorizer sauvegardé : models/tfidf_vectorizer.joblib")

    return model, vectorizer


if __name__ == "__main__":
    train_tfidf()