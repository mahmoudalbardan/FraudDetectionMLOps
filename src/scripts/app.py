from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('fraud_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    test_data =  np.array(request.json).reshape(1, -1) # given that data is in json format
    prediction = 1 if model.predict(test_data) == -1 else 0
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
