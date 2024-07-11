import pickle
from flask import Flask, request, render_template

import pandas as pd
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open("scaler.pkl","rb"))
regpredict = pickle.load(open("RegPredict.pkl","rb"))

@app.route('/')

def home():
    return render_template("home.html")

@app.route('/predict', methods= ["POST"])

def predict():
    data=scaler.transform(np.array([float(x) for x in request.form.values()]).reshape(1,-1))
    result_data= regpredict.predict(data)[0]
    return render_template("home.html", answer= "House_Price : {}".format(result_data))


if __name__== "__main__":
    app.run(debug=True)