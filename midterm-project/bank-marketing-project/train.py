"""Train the final bank term-deposit model and save it to model.bin.

Usage:
    python train.py

Reads data/bank-full.csv (see README for download instructions), trains the
final model with the parameters chosen in notebook.ipynb, prints validation
and test ROC AUC, and saves (DictVectorizer, model) to model.bin.
"""
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_PATH = "data/bank-full.csv"
OUTPUT_FILE = "model.bin"

# >>> Update these with the best parameters found in notebook.ipynb <<<
XGB_PARAMS = {
    "eta": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 4,
    "seed": 1,
}
NUM_ROUNDS = 200

CATEGORICAL = ["job", "marital", "education", "default", "housing",
               "loan", "contact", "month", "poutcome"]
NUMERICAL = ["age", "balance", "day", "campaign", "pdays", "previous"]


def load_data(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    # 'duration' is only known after the call ends, so it would leak the answer
    df = df.drop(columns=["duration"])
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def train(df_train, y_train, params=XGB_PARAMS, rounds=NUM_ROUNDS):
    dicts = df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    dtrain = xgb.DMatrix(X, label=y_train, feature_names=list(dv.get_feature_names_out()))
    model = xgb.train(params, dtrain, num_boost_round=rounds)
    return dv, model


def predict(df, dv, model):
    dicts = df[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    X = dv.transform(dicts)
    d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    return model.predict(d)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df["y"])
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 5-fold cross-validation as a sanity check
    print("Cross-validation (5 folds)")
    scores = []
    for fold, (train_idx, val_idx) in enumerate(KFold(5, shuffle=True, random_state=1).split(df_full_train)):
        tr, va = df_full_train.iloc[train_idx], df_full_train.iloc[val_idx]
        dv, model = train(tr, tr["y"].values)
        auc = roc_auc_score(va["y"], predict(va, dv, model))
        scores.append(auc)
        print(f"  fold {fold}: AUC = {auc:.3f}")
    print(f"  mean AUC = {sum(scores) / len(scores):.3f}")

    # final model on the whole training part, evaluated on the held-out test set
    print("Training the final model...")
    dv, model = train(df_full_train, df_full_train["y"].values)
    test_auc = roc_auc_score(df_test["y"], predict(df_test, dv, model))
    print(f"Test AUC = {test_auc:.3f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {OUTPUT_FILE}")
