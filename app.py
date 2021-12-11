import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from sklearn import linear_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow_hub as tfhub

app = Flask(__name__)
model = load_model("deeplearning.h5", custom_objects={
                       'KerasLayer': tfhub.KerasLayer})



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    wilayah = str(request.form['wilayah'])
    waktu = str(request.form['waktu'])
    kelembaban_persen = str(request.form['kelembaban_persen'])
    suhu_derajat_celsius = str(request.form['suhu_derajat_celsius'])
    banyakkotarawan = int(request.form['banyakkotarawan'])
    banyakkotarawan = (banyakkotarawan) // 25
    items = ['wilayah', 'waktu', 'kelembaban_persen', 'suhu_derajat_celsius', 'banyakkotarawan']
    data = [[wilayah, waktu, kelembaban_persen, suhu_derajat_celsius, banyakkotarawan]]
    data_df = pd.DataFrame(data=data, columns=items)

    col_cat = [x for x in data_df.columns if x not in ["BanyakKotaRawan"]]
    for var in col_cat:      
        cat_list = 'var'+'_'+var
        cat_list = pd.get_dummies(data_df[var], prefix=var)
        data1= data_df.join(cat_list)
        data_df=data1

    data_df = [np.array(data_df)]
    prediction = model.predict(data_df)
    output = prediction
    

    return render_template("index.html", prediction_text='Cuaca yang akan terjadi nanti yaitu {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)