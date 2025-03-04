import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_preprocessing import load_tsv_files, clean_data
from feature_engineering import encode_features, create_target

def prepare_data(dir):
    df = load_tsv_files(dir)
    df = clean_data(df)

    features, _ = encode_features(df)
    target = create_target(df)

    return features, target

def train_model(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier.fit(X_train, y_train)
    return classifier

if __name__ == "__main__":
    dir = "path"            # <--- replace with data directory

    X, y = prepare_data(dir)
    X_train, X_test, y_train, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    

