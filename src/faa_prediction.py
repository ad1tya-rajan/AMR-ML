import os
import torch
import pandas as pd
import numpy as np
import joblib

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from data_preprocessing import parse_fasta, parse_input_fasta
from model_dev.transformer_model import AMRClassifier
from amr_dataset import AMRDataset

CONFIDENCE_THRESHOLD = 0.90             # we might make this higher in the future to reduce false positives

def predict_from_faa(fasta_path, output_path):

    print("Parsing FASTA file...")
    df = parse_input_fasta(fasta_path)
    df = df[df["sequence"].notna() & (df["sequence"].str.strip() != "")]
    sequences = [" ".join(list(seq)) for seq in df["sequence"]]

    print("Loading tokenizer, model, and label encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    le = joblib.load("../models/transformer/label_encoder.joblib")

    model = AMRClassifier("Rostlab/prot_bert", num_classes=len(le.classes_))
    model.load_state_dict(torch.load("../models/transformer/amr_transformer_model.pt"))
    model.to(device)
    model.eval()

    dataset = AMRDataset(sequences, labels=None, tokenizer_name="Rostlab/prot_bert", max_length=512)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_confidences = []

    print("Running inference...")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())

    df["Confidence"] = all_confidences
    df["Predicted Class"] = [
        le.inverse_transform([pred])[0] if conf >= CONFIDENCE_THRESHOLD else "No_AMR_Detected"
        for pred, conf in zip(all_preds, all_confidences)
    ]

    df_filtered = df[df["Predicted Class"] != "No_AMR_Detected"]

    print(f"Saving results to {output_path}...")
    df_filtered[["gene_name", "Predicted Class", "Confidence"]].to_csv(output_path, sep="\t", index=False)
    print("AMR prediction complete! Results saved to", output_path)

if __name__ == "__main__":
    fasta_file = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/raw/PATRIC2.faa"  # Replace with your input path
    output_tsv = "results/PATRIC2_amr_predictions.tsv"
    predict_from_faa(fasta_file, output_tsv)
