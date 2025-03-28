import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.utils import class_weight
import numpy as np

# from training import class_weights

def build_xgb_model(params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "objective": "multi:softmax",
            "learning_rate": 0.1,
            "num_class": 12,
            "max_depth": 6,
            "tree_method": "hist",
            "device": "cuda"
        }

    model = XGBClassifier(**params)
    return model
    
def train_xgb_model(model, X_train, y_train):

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # sample_weights = class_weights[y_train]
    model.fit(X_train, y_train)
    return model