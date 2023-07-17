import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# gcloud config set project credit-score-model


@app.route("/")
def Home():
    return jsonify({"ready": True})


@app.route("/predict/<loanId>", methods=["POST"])
def predict(loanId):
    content = request.json
    float_features = pd.DataFrame(content)
    prediction = model.predict(float_features)
    return jsonify({"credit_score": str(prediction[0]), "loanId": loanId})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
