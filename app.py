import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template ,send_file,make_response,Response,flash,redirect,url_for
import joblib
import xgboost

app = Flask(__name__)

model = joblib.load('./model.pkl')

x_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# index
@app.route('/')
def home():
    return render_template('index.html')

# predict
@app.route('/predict',methods=['POST'])
def predict():
    features = np.array([float(x) for x in request.form.values()]).reshape(1,-1)
    features = pd.DataFrame(features,columns=x_cols)
    preds = model.predict(features)[0]
    preds = round(preds,2)
    return render_template('index.html',output='predict is:{}'.format(preds))

# run app
if __name__ == "__main__":
    app.run(debug = True)