import re
import string
import yaml as yml
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


def load_config(config_path: str = "configs/config.yml") -> dict:
    with open(config_path, "r") as f:
        return yml.safe_load(f)


def clean_text_tfidf(text: str) -> str:
    """Nettoyage complet pour TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def clean_text_bert(text: str) -> str:
    """Nettoyage minimal pour BERT — conserve le contexte."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(config_path: str = "configs/config.yml"):
    """Pipeline : emails_clean.csv → nettoyage NLP → train.csv / test.csv"""
    config = load_config(config_path)

    # Chargement
    df = pd.read_csv(config["data"]["processed_path"])
    print(f"Dataset chargé : {df.shape}")

    # Nettoyage NLP
    df["text_tfidf"] = df["text"].apply(clean_text_tfidf)
    df["text_bert"]  = df["text"].apply(clean_text_bert)

    # Suppression textes vides après nettoyage
    before = len(df)
    df = df[df["text_tfidf"].str.strip() != ""].reset_index(drop=True)
    print(f"Supprimés après nettoyage TF-IDF : {before - len(df)}")

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text_tfidf", "text_bert"]],
        df["label"],
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=df["label"]
    )

    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"Distribution train → {y_train.value_counts().to_dict()}")

    # Sauvegarde
    train_df = X_train.copy()
    train_df["label"] = y_train.values
    test_df = X_test.copy()
    test_df["label"] = y_test.values

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Sauvegardé : data/processed/train.csv et test.csv")
    return train_df, test_df


if __name__ == "__main__":
    preprocess()