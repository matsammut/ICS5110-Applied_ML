from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model.joblib')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expect JSON input
    input_features = np.array(data['features']).reshape(1, -1)  # Reshape for model
    prediction = model.predict(input_features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)