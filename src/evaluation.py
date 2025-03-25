from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

from training import load_data, prepare_train_test_data, aro_id_to_gene_name

def evaluate_model(model, X_test, y_test, le, aro_to_gene):

    # prediction

    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    predicted_aro = le.inverse_transform(y_pred)
    predicted_gene = [aro_to_gene.get(aro, "Unknown") for aro in predicted_aro]

    print("Predicted ARO IDs:\n", predicted_aro)
    print("Predicted Gene Names:\n", predicted_gene)

    # evaluation

    report = classification_report(y_test, y_pred, target_names = le.classes_)      # skl classification report
    print("Classification Report:\n", report)

    f1_macro = f1_score(y_test, y_pred, average="macro")                            # skl f1 score
    print("F1 Score (macro):", f1_macro)

    classes = np.arange(len(le.classes_))
    y_test_binary = label_binarize(y_test, classes=classes)
    roc_auc_score = roc_auc_score(y_test_binary, y_prob, multi_class="ovr", average="macro")
    print("ROC AUC Score:", roc_auc_score)                                          # skl roc auc score

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")                                                    # confusion matrix
    plt.show()

def main():

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    le = joblib.load("models/xgboost/label_encoder.pkl")
    model = joblib.load("models/xgboost/xgb_model.pkl")

    fasta_file = "path/to/fasta/file"
    df, vocab = load_data(fasta_file)

    print("Data loaded and processed. DF shape: ", df.shape)
    aro_to_gene = aro_id_to_gene_name(df)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, le, aro_to_gene)

if __name__ == "__main__":
    main()