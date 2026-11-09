"""Flask web service that serves the term-deposit model.

Run locally:   python predict.py
Run in Docker: see README (gunicorn serves predict:app on port 9696)
"""
import pickle

import xgboost as xgb
from flask import Flask, jsonify, request

MODEL_FILE = "model.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("bank-marketing")


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    if not isinstance(customer, dict):
        return jsonify({"error": "send one customer as a JSON object"}), 400

    X = dv.transform([customer])
    d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    prob = float(model.predict(d)[0])

    return jsonify({
        "subscribe_probability": prob,
        "subscribe": bool(prob >= 0.5),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
