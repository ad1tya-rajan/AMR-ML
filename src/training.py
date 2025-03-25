import json
import os
import pandas as pd
import numpy as np   
import joblib                                  # type: ignore  
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.preprocessing import LabelEncoder
from models.xgboost import train_xgb_model
from models.xgboost import build_xgb_model

from data_preprocessing import parse_fasta
from feature_engineering import kmer_feature_vector, comp_vector, build_kmer_vocab

def get_aro_ontology(json_path):
    with open(json_path, "r") as f:
        aro_ontology = json.load(f)

    aro_to_class = {}

    for entry in aro_ontology.values():

        if not isinstance(entry, dict):
            continue

        aro_accession = entry.get("ARO_accession", "").strip()
        if not aro_accession:
            continue

        aro_id = "ARO:" + aro_accession
        drug_classes = []
        categories = entry.get("ARO_category", [])

        if isinstance(categories, dict):
            categories = list(categories.values())
        elif not isinstance(categories, list):
            categories = []

        for cat_data in categories:
            if isinstance(cat_data, dict) and cat_data.get("category_aro_class_name") == "Drug Class":
                name = cat_data.get("category_aro_name", "")
                if name:
                    drug_classes.append(name)

        aro_to_class[aro_id] = drug_classes[0] if drug_classes else "Unknown"

    return aro_to_class

def load_data(fasta_file, json_path, label_col = "drug_class"):
    df = parse_fasta(fasta_file)

    # get ARO ontology from json file
    aro_to_class = get_aro_ontology(json_path)
    df["drug_class"] = df["aro_id"].map(aro_to_class).fillna("Unknown")

    # feature engineering
    df["comp_vector"] = df["sequence"].apply(comp_vector)
    sequences = df["sequence"].tolist()
    vocab = build_kmer_vocab(sequences, k = 3, min_count=1)
    df["kmer_vector"] = df["sequence"].apply(lambda seq: kmer_feature_vector(seq, k = 3, vocab = vocab))

    # concatenate comp_vector and kmer_vector
    df["features"] = df.apply(lambda row: np.concatenate((row["comp_vector"], row["kmer_vector"])), axis = 1)
    df["label"] = df[label_col]

    return df, vocab

def prepare_train_test_data(df, test_size = 0.2, random_state = 42):
    X = np.stack(df["features"].values)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ---- debug ----

    print("Unique encoded labels:", np.unique(y))
    print("Number of classes:", len(le.classes_))
    print("Label encoder classes:", le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    save_dir = os.path.join("..", "data", "processed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    joblib.dump(le, os.path.join("models", "xgboost", "label_encoder.joblib"))

    return X_train, X_test, y_train, y_test, le

def aro_id_to_gene_name(df):
    mapping = {}

    for aro_id, group in df.groupby("aro_id"):
        gene_name = group["gene_name"].value_counts().idxmax()
        mapping[aro_id] = gene_name

    return mapping

def main():
    fasta_file = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/raw/protein_fasta_protein_homolog_model.fasta"
    json_path = "/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/external/card.json"
    df, vocab = load_data(fasta_file, json_path)
    aro_to_gene = aro_id_to_gene_name(df)

    print("Data loaded and processed. DF shape: ", df.shape)

    X_train, X_test, y_train, y_test, le = prepare_train_test_data(df)

    num_classes = len(le.classes_)

    params = {
        "n_estimators": 100,
        "objective": "multi:softmax",
        "learning_rate": 0.1,
        "num_class": num_classes,
        "max_depth": 6,
        "tree_method": "hist",
        "device": "cuda"
    }

    model = build_xgb_model(params)
    model = train_xgb_model(model, X_train, y_train)

    print("Baseline model (XGBoost) training complete!")

    model_dir = os.path.join("..", "models", "xgboost")
    os.makedirs(model_dir, exist_ok=True)
    
    
    model_save_path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(model, model_save_path)
    print("Model saved to", model_save_path)

if __name__ == "__main__":
    main()