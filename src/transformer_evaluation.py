import torch
import joblib
from model_dev.transformer_model import AMRClassifier
from amr_dataset import AMRDataset
from torch.utils.data import DataLoader
import pandas as pd
from data_preprocessing import parse_fasta
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_data():
    le = joblib.load("../models/transformer/label_encoder.joblib")
    df = parse_fasta("../data/raw/AMRProt.fa")
    df = df[df["drug_class"].notna() & (df["drug_class"].str.strip() != "")]
    df = df[df["drug_class"].isin(le.classes_)]  # Keep only known classes
    df["label"] = le.transform(df["drug_class"])

    sequences = [" ".join(list(seq)) for seq in df["sequence"]]
    labels = df["label"].tolist()

    dataset = AMRDataset(sequences, labels)
    return dataset, le

def evaluate_model(model, dataloader, le):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = [le.inverse_transform([pred])[0] for pred in all_preds]
    all_labels = [le.inverse_transform([label])[0] for label in all_labels]

    print("All predictions:", all_preds)
    print("All labels:", all_labels)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    print("F1 Score:", f1_score(all_labels, all_preds, average='weighted'))

    # conf_mat = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_mat, xticklabels=le.classes_, yticklabels=le.classes_, annot=False, cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # plt.show()

def main():
    dataset, le = load_model_data()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = AMRClassifier("Rostlab/prot_bert", num_classes=len(le.classes_))
    model.load_state_dict(torch.load("../models/transformer/amr_transformer_model.pt"))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    evaluate_model(model, dataloader, le)

if __name__ == "__main__":  
    main()