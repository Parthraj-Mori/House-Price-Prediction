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
    result_data= regpredict.predict(data)[0]*83
    
   # Description info
    data_1= request.form["bedrooms"]
    data_2=request.form["bathrooms"]
    data_3= request.form["square_footage"]
    data_4= 2024-int(request.form["age"])
    
    a=[result_data,data_1,data_2,data_3]

    return render_template("index.html", answer= "{}".format(int(result_data)), bedroom ="{}".format(int(data_1)),bathroom ="{}".format(int(data_2)),square_fit ="{}".format(int(data_3)),year="{}".format(int(data_4)))
if __name__== "__main__":
    app.run(debug=True)