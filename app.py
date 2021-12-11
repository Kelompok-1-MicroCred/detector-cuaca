import numpy as np
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


app = Flask(__name__)
model = load_model('deeplearning.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction

    wilayah = str(request.form['wilayah'])
    waktu = str(request.form['waktu'])
    kelembaban_persen = str(request.form['kelembaban_persen'])
    suhu_derajat_celsius = str(request.form['suhu_derajat_celsius'])
    banyakkotarawan = str(request.form['banyakkotarawan'])

    data = [wilayah, waktu, kelembaban_persen, suhu_derajat_celsius, banyakkotarawan]

    data_df = pd.DataFrame(data=data, columns=items)

    return render_template("index.html", prediction_text='Cuaca yang akan terjadi nanti yaitu {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)