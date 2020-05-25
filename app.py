from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import math

app = Flask(__name__)

model = pickle.load(open('taxi.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])

def predict():
    features = [int(value) for value in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text = 'Number of Weekly Rides Should be {}'.format(math.floor(output)))


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
