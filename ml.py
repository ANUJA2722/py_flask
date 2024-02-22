from flask import Flask,render_template, request
import numpy as np
import pickle


model = pickle.load(open('crop.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html' )

@app.route('/pred')
def pred():
    return render_template('prediction.html' )

@app.route('/predict',methods=['POST'])
def predict():
    data1 = request.form['N']
    data2 = request.form['P']
    data3 = request.form['K']
    data4 = request.form['temp']
    data5 = request.form['humidity']
    data6 = request.form['ph']
    data7 = request.form['rainfall']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7]])
    pred = model.predict(arr)
    return render_template('prediction.html', data=pred)

if __name__ == "__main__":
    app.run()