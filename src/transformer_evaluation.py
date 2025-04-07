# TODO: add class weighting to balance the dataset (consider SMOTE or other methods)

import os
import numpy as np
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

metrics = {
    "accuracy": [],
    "f1_score_macro": [],
    "f1_score_weighted": [],
}

def load_model_data():
    le = joblib.load("../models/transformer/label_encoder.joblib")
    val_indices = torch.load("../models/transformer/val_indices.pt", weights_only=False)
    # train_indices = torch.load("../models/transformer/train_dataset.pt")

    df = parse_fasta("../data/raw/AMRProt.fa")
    df = df[df["drug_class"].notna() & (df["drug_class"].str.strip() != "")]
    df = df[df["drug_class"].isin(le.classes_)]  # Keep only known classes
    df["label"] = le.transform(df["drug_class"])

    sequences = [" ".join(list(seq)) for seq in df["sequence"]]
    labels = df["label"].tolist()

    dataset = AMRDataset(sequences, labels)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    return DataLoader(val_subset, batch_size=8), le

def evaluate_model(model, dataloader, le, output_path=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # all_preds = [le.inverse_transform([pred])[0] for pred in all_preds]
    # all_labels = [le.inverse_transform([label])[0] for label in all_labels]

    # print("All predictions:", all_preds)
    # print("All labels:", all_labels)

    actual_labels = sorted(set(all_labels + all_preds))  # All used label indices
    actual_classes = le.inverse_transform(actual_labels)  # Their string names

    print("Classification Report:")
    report = classification_report(
        all_labels, 
        all_preds,
        labels=actual_labels, 
        target_names=actual_classes, 
        zero_division=0, 
        output_dict=True)
    
    report_df = pd.DataFrame(report).T
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    print(report_df)
    f1_score_macro = f1_score(all_labels, all_preds, average='macro')
    f1_score_weighted = f1_score(all_labels, all_preds, average='weighted')

    print("F1 Score (macro):", f1_score_macro)
    print("F1 Score (weighted):", f1_score_weighted)
    print("Accuracy: ", accuracy)

    metrics["accuracy"].append(accuracy)
    metrics["f1_score_macro"].append(f1_score_macro)
    metrics["f1_score_weighted"].append(f1_score_weighted)

    print("Metrics:", metrics)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report_df.index.name = "Drug Class"
        report_df.to_csv(output_path, sep="\t")
        print(f"Classification report saved to {output_path}")

    # conf_mat = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_mat, xticklabels=le.classes_, yticklabels=le.classes_, annot=False, cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # plt.show()

def plot_metrics(metrics, output_path=None):

    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        if len(values) > 0:  # Ensure the list is not empty
            plt.plot(range(len(values)), values, label=metric_name)
        else:
            print(f"Warning: No data for metric '{metric_name}'")
    
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Metrics plot saved to {output_path}")

    # plt.show()
    plt.close()

def main():
    dataloader, le = load_model_data()

    model = AMRClassifier("Rostlab/prot_bert", num_classes=len(le.classes_))
    model.load_state_dict(torch.load("../models/transformer/amr_transformer_model.pt"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    evaluate_model(model, dataloader, le, output_path=None)
    plot_metrics(metrics, output_path="results/transformer_metrics.png")

if __name__ == "__main__":  
    main()