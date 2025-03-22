from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

from training import load_data, prepare_train_test_data

def evaluate_model(model, X_test, y_test, le):

    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_test, y_pred, target_names = le.classes_)
    print("Classification Report:\n", report)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    print("F1 Score (macro):", f1_macro)

    classes = np.arange(len(le.classes_))
    y_test_binary = label_binarize(y_test, classes=classes)
    roc_auc_score = roc_auc_score(y_test_binary, y_prob, multi_class="ovr", average="macro")
    print("ROC AUC Score:", roc_auc_score)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():

    fasta_file = "path/to/fasta/file"
    df, vocab = load_data(fasta_file)

    print("Data loaded and processed. DF shape: ", df.shape)

    X_train, X_test, y_train, y_test, le = prepare_train_test_data(df)

    model_file = "path/to/xgb_model.joblib"

    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print("Model loaded from", model_file)
        evaluate_model(model, X_test, y_test, le)

    else:
        print("Model not found at", model_file)

if __name__ == "__main__":
    main()