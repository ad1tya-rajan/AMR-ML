import pandas as pd # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore

def encode_features(df):
    numerical_features = df[["percent_identity", "coverage"]].copy()

    categorical_features = ["antibiotic_class"]
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_features))

    features = pd.concat([numerical_features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return features, encoder

def create_target(df, target_column = "antibiotic_class"):
    return df[target_column]

if __name__ == "__main__":
    sample_df = pd.DataFrame({
        # define dummy data
    })

features, encoder = encode_features(sample_df)
target = create_target(sample_df)

print("Features: \n", features.head())
print("Features: \n", target.head())
