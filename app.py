import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

#defining a flask app
app=Flask(__name__)
reg_model=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)

    output = reg_model.predict(final_input)[0]
    return render_template("home.html",prediction_text = "Sales prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
    