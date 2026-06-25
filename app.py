from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/HDI.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["LifeExpectancy"]),
        float(request.form["MeanYearsSchooling"]),
        float(request.form["ExpectedYearsSchooling"]),
        float(request.form["GNIPerCapita"])
    ]

    prediction = model.predict([features])[0]

    return render_template("result.html", prediction=round(prediction, 3))

if __name__ == "__main__":
    app.run(debug=True)