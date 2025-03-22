import xgboost as xgb
from xgboost import XGBClassifier

def build_xgb_model(params = None):
    if params is None:
        params = {
            "n_estimators": 100,
            "objective": "multi:softmax",
            "learning_rate": 0.1,
            "num_class": 12,
            "max_depth": 6
        }

        model = XGBClassifier(**params)
        return model
    
def train_xgb_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model