from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    size = data.get('size')
    if not size:
        return jsonify({'message': 'There was an error'}), 400
    X_new = np.array([[size]])
    prediction = model.predict(X_new)

    return jsonify({"price": round(prediction[0], 2)})

if __name__ == '__main__':
    app.run()