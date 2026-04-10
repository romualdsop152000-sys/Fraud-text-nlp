import yaml as yml
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, average_precision_score


def load_config(config_path: str = "configs/config.yml") -> dict:
    with open(config_path, "r") as f:
        return yml.safe_load(f)


class EmailDataset(Dataset):
    """Dataset PyTorch pour les emails."""
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


def train_bert(config_path: str = "configs/config.yml", epochs: int = 3, batch_size: int = 16):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Chargement des données
    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    X_train = train_df["text_bert"].fillna("")
    y_train = train_df["label"]
    X_test  = test_df["text_bert"].fillna("")
    y_test  = test_df["label"]

    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    print("Tokenisation en cours...")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    test_dataset  = EmailDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    print(f"Batches train : {len(train_loader)} | Batches test : {len(test_loader)}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Entraînement avec logs par batch
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Log tous les 50 batches
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} — Loss: {avg:.4f}")

        print(f"Epoch {epoch + 1} terminée — Loss moyenne : {total_loss / len(train_loader):.4f}")

    # Évaluation
    print("\nÉvaluation sur le test set...")
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds   = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_proba.extend(probs)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Évaluation batch {batch_idx + 1}/{len(test_loader)}")

    print("\n=== Résultats DistilBERT ===")
    print(classification_report(all_labels, all_preds, target_names=["Non-Fraude", "Fraude"]))
    print(f"F1 (fraude)  : {f1_score(all_labels, all_preds):.4f}")
    print(f"PR-AUC       : {average_precision_score(all_labels, all_proba):.4f}")

    # Sauvegarde — après l'évaluation complète
    os.makedirs("models/bert", exist_ok=True)
    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")
    print("\nModèle BERT sauvegardé : models/bert/")

    return model, tokenizer


if __name__ == "__main__":
    train_bert()