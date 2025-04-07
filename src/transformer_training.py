import os
import gc
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocessing import parse_fasta
from model_dev.transformer_model import AMRClassifier
from amr_dataset import AMRDataset

def load_data(fasta_file, label_col = "drug_class"):
    df = parse_fasta(fasta_file)

    df[label_col] = df[label_col].fillna("Unknown")
    df = df[df[label_col].str.strip() != ""]

    return df

def prepare_label_encoder(df, label_col = "drug_class"):
    label_counts = Counter(df[label_col])

    df = df[df[label_col].map(label_counts) > 1]  # Filter out classes with only one sample

    le = LabelEncoder()
    df.loc[:, label_col] = le.fit_transform(df[label_col])

    os.makedirs("../models/transformer", exist_ok=True)
    joblib.dump(le, "../models/transformer/label_encoder.joblib")
    print("Label encoder saved to ../models/transformer/label_encoder.joblib")
    return df, le

torch.cuda.empty_cache()
gc.collect()

def train_transformer_model(df, le, label_col = "drug_class"):

    sequences = [" ".join(list(seq)) for seq in df["sequence"].tolist()]
    labels = df[label_col].tolist()

    dataset = AMRDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    torch.save(train_dataset, "../models/transformer/train_dataset.pt")
    torch.save(val_dataset.indices, "../models/transformer/val_indices.pt")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMRClassifier("Rostlab/prot_bert", num_classes=len(le.classes_)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("Training transformer model...")

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")
    model_save_path = "../models/transformer/amr_transformer_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    fasta_file = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/raw/AMRProt.fa"  # Path to your FASTA file
    label_col = "drug_class"  # Column name for labels

    df = load_data(fasta_file, label_col)
    df, le = prepare_label_encoder(df, label_col)
    train_transformer_model(df, le, label_col)

if __name__ == "__main__":
    main()