from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


linear_model = pickle.load(open("models\linear_regression_model.pkl", "rb"))
scaler_model = pickle.load(open("models\scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
        if request.method == "POST":
            weight = float(request.form["weight"])
            new_data_scaled= scaler_model.transform([[weight]])
            prediction = linear_model.predict(new_data_scaled)
            
            return render_template("home.html", result=prediction[0])
            
            
        else :
            return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    