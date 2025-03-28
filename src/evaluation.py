from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import os

def evaluate_model(model, X_test, y_test, le, output_csv=None):
    # Predict probabilities and labels
    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # Check all unique label indices in your test set
    print("DEBUG: \n")
    print("Unique y_test labels:", np.unique(y_test))
    print("Unique y_pred labels:", np.unique(y_pred))
    print("Label encoder classes:", le.classes_)
    print("Number of classes:", len(le.classes_))

    # Inverse transform to get class names
    predicted_labels = le.inverse_transform(y_pred)
    print("Predicted Drug Classes:\n", predicted_labels)

    # Evaluation Metrics
    # print("Classification Report:\n", classification_report(
    #     y_test, y_pred,
    #     labels=np.arange(len(le.classes_)),
    #     target_names=le.classes_,
    #     zero_division=0
    # ))

    print("Classification Report (filtered zero support classes):")
    report_dict = classification_report(y_test, y_pred, 
                                        labels=np.arange(len(le.classes_)),
                                        target_names=le.classes_, 
                                        zero_division=0, 
                                        output_dict=True)
    
    for label, scores in report_dict.items():
        if isinstance(scores, dict) and label == "":
            print(f"blank label: {scores}")
    
    filtered = {k: v for k, v in report_dict.items() if isinstance(v, dict) and v['support'] > 0}
    report_df = pd.DataFrame(filtered).T
    report_df.index.name = "Drug Class"
    print(report_df)

    print("F1 Score (macro):", f1_score(y_test, y_pred, average="macro"))

    if output_csv:
        # Save the classification report to a CSV file
        report_df = pd.DataFrame(report_dict).T
        report_df.to_csv(output_csv)
        print(f"Classification report saved to {output_csv}")

    # y_test_binary = label_binarize(y_test, classes=np.arange(len(le.classes_)))
    # try:
    #     roc_auc = roc_auc_score(y_test_binary, y_prob, multi_class="ovr", average="macro")
    #     print("ROC AUC Score:", roc_auc)
    # except ValueError as e:
    #     print("ROC AUC could not be computed:", e)

    # Confusion matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(conf_matrix, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # os.makedirs("results", exist_ok=True)
    # plt.savefig("results/confusion_matrix.png", dpi=300)
    # plt.close()
    # print("Confusion matrix saved to results/confusion_matrix.png")


def main():
    # Load saved test data and model
    X_test = np.load("/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/processed/X_test.npy")
    y_test = np.load("/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/data/processed/y_test.npy")
    le = joblib.load("/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/models/xgboost/label_encoder.joblib")
    model = joblib.load("/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/models/xgboost/xgb_model.pkl")

    print("Evaluating model on held-out test set...")
    evaluate_model(model, X_test, y_test, le, output_csv="/home/cvm-alamlab/Desktop/Aditya/AMR_Project/AMR-ML/src/results/classification_report.csv")

if __name__ == "__main__":
    main()
