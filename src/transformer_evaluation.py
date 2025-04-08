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
    le_super = joblib.load("../models/transformer/label_encoder_super.joblib")
    le_sub = joblib.load("../models/transformer/label_encoder_sub.joblib")

    df = parse_fasta("../data/raw/AMRProt.fa")
    df = df[df["super_class"].isin(le_super.classes_)]
    df = df[df["drug_class"].isin(le_sub.classes_)]

    df["super_label"] = le_super.transform(df["super_class"])
    df["sub_label"] = le_sub.transform(df["drug_class"])

    sequences = [" ".join(list(seq)) for seq in df["sequence"]]
    super_labels = df["super_label"].tolist()
    sub_labels = df["sub_label"].tolist()


    dataset = AMRDataset(sequences, super_labels, sub_labels)
    val_indices = torch.load("../models/transformer/val_indices.pt")
    val_dataset = torch.utils.Subset(dataset, val_indices)

    return DataLoader(val_dataset, batch_size=8), le_super, le_sub

def evaluate_model(model, dataloader, le_super, le_sub, output_path=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_super, pred_super = [], []
    true_sub, pred_sub = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            super_labels = batch["super_labels"].to(device)
            sub_labels = batch["sub_labels"].to(device)

            super_logits, sub_logits = model(input_ids, attention_mask)
            super_preds = torch.argmax(super_logits, dim=-1)
            sub_preds = torch.argmax(sub_logits, dim=-1)

            true_super.extend(super_labels.cpu().numpy())
            pred_super.extend(super_preds.cpu().numpy())
            true_sub.extend(sub_labels.cpu().numpy())
            pred_sub.extend(sub_preds.cpu().numpy())

    print("Super Class Report:")
    print(classification_report(true_super, pred_super, target_names=le_super.classes_))
    print("Super F1:", f1_score(true_super, pred_super, average='macro'))

    print("Sub Class Report:")
    print(classification_report(true_sub, pred_sub, target_names=le_sub.classes_))
    print("Sub F1:", f1_score(true_sub, pred_sub, average='macro'))

    # print("Classification Report:")
    # report = classification_report(
    #     all_labels, 
    #     all_preds,
    #     labels=actual_labels, 
    #     target_names=actual_classes, 
    #     zero_division=0, 
    #     output_dict=True)
    
    # report_df = pd.DataFrame(report).T
    # accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    # print(report_df)
    # f1_score_macro = f1_score(all_labels, all_preds, average='macro')
    # f1_score_weighted = f1_score(all_labels, all_preds, average='weighted')

    # print("F1 Score (macro):", f1_score_macro)
    # print("F1 Score (weighted):", f1_score_weighted)
    # print("Accuracy: ", accuracy)

    # metrics["accuracy"].append(accuracy)
    # metrics["f1_score_macro"].append(f1_score_macro)
    # metrics["f1_score_weighted"].append(f1_score_weighted)

    # print("Metrics:", metrics)
    
    # if output_path:
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     report_df.index.name = "Drug Class"
    #     report_df.to_csv(output_path, sep="\t")
    #     print(f"Classification report saved to {output_path}")

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
    dataloader, le_super, le_sub = load_model_data()

    model = AMRClassifier("Rostlab/prot_bert", num_super_classes=len(le_super.classes_), num_sub_classes=len(le_sub.classes_))
    model.load_state_dict(torch.load("../models/transformer/amr_transformer_model.pt"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    evaluate_model(model, dataloader, le_super, le_sub, output_path=None)
    plot_metrics(metrics, output_path="results/transformer_metrics.png")

if __name__ == "__main__":  
    main()