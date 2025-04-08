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

def load_data(fasta_file):
    df = parse_fasta(fasta_file)

    df = df[df["drug_class"].notna() & df["super_class"].notna()]
    df = df[(df["drug_class"].str.strip() != "") & (df["super_class"].str.strip() != "")]

    return df

def prepare_label_encoder(df):
    # Filter rare sub-classes (you can also do this per super_class if needed)
    sub_counts = Counter(df["drug_class"])
    df = df[df["drug_class"].map(sub_counts) > 1]

    # Fit encoders
    le_super = LabelEncoder()
    le_sub = LabelEncoder()
    df["super_label"] = le_super.fit_transform(df["super_class"])
    df["sub_label"] = le_sub.fit_transform(df["drug_class"])

    # Save encoders
    model_dir = "../models/transformer"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(le_super, os.path.join(model_dir, "label_encoder_super.joblib"))
    joblib.dump(le_sub, os.path.join(model_dir, "label_encoder_sub.joblib"))
    print("Saved super and sub-class label encoders.")

    return df, le_super, le_sub

torch.cuda.empty_cache()
gc.collect()

def train_transformer_model(df, le_super, le_sub):

    sequences = [" ".join(list(seq)) for seq in df["sequence"].tolist()]
    super_labels = df["super_label"].values
    sub_labels = df["sub_label"].values

    dataset = AMRDataset(sequences, super_labels, sub_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    torch.save(train_dataset, "../models/transformer/train_dataset.pt")
    torch.save(val_dataset.indices, "../models/transformer/val_indices.pt")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMRClassifier("Rostlab/prot_bert", num_super_classes=len(le_super.classes_), num_sub_classes=len(le_sub.classes_)).to(device)

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
            super_labels = batch["super_labels"].to(device)
            sub_labels = batch["sub_labels"].to(device)

            optimizer.zero_grad()

            super_logits, sub_logits = model(input_ids, attention_mask)
            loss_super = criterion(super_logits, super_labels)
            loss_sub = criterion(sub_logits, sub_labels)
            loss = loss_super + loss_sub

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

    df = load_data(fasta_file)
    df, le_super, le_sub = prepare_label_encoder(df)
    train_transformer_model(df, le_super, le_sub)

if __name__ == "__main__":
    main()