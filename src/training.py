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

def load_data(fasta_file):
    df = parse_fasta(fasta_file)

    df["comp_vector"] = df["sequence"].apply(comp_vector)

    sequences = df["sequence"].tolist()
    vocab = build_kmer_vocab(sequences, k = 3, min_count=1)
    df["kmer_vector"] = df["sequence"].apply(lambda seq: kmer_feature_vector(seq, k = 3, vocab = vocab))

    df["features"] = df.apply(lambda row: np.concatenate((row["comp_vector"], row["kmer_vector"])), axis = 1)
    df["label"] = df["gene_name"]

    return df, vocab

def prepare_train_test_data(df, test_size = 0.2, random_state = 42):
    X = np.stack(df["features"].values)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test, le

def main():
    fasta_file = "path/to/fasta/file"
    df, vocab = load_data(fasta_file)

    print("Data loaded and processed. DF shape: ", df.shape)

    X_train, X_test, y_train, y_test, le = prepare_train_test_data(df)

    num_classes = len(le.classes_)

    params = {
        "n_estimators": 100,
        "objective": "multi:softmax",
        "learning_rate": 0.1,
        "num_class": num_classes,
        "max_depth": 6
    }

    model = build_xgb_model(params)
    model = train_xgb_model(model, X_train, y_train)

    print("Baseline model (XGBoost) training complete!")

    model_file = "path/to/xgb_model.joblib"

    if not os.path.exists(model_file):
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
    
    model_save_path = os.path.join(model_file, "xgb_model.joblib")
    joblib.dump(model, model_save_path)
    print("Model saved to", model_save_path)

if __name__ == "__main__":
    main()