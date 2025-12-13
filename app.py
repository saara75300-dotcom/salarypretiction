
%%writefile app.py
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
# Make sure the model file 'Salary_pre_linear_reg_model.pkl' is in the same directory
with open('Salary_pre_linear_reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Salary Prediction API! Use /predict to get salary predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    gender = data.get('gender') # Expects 'Male' or 'Female'

    if gender is None:
        return jsonify({'error': 'Please provide a gender (Male or Female).'}), 400

    # Convert gender to numerical representation (as used during training)
    # Assuming 'Gender_Male' is the column name after one-hot encoding
    if gender.lower() == 'male':
        prediction_input = pd.DataFrame([[1]], columns=['Gender_Male'])
    elif gender.lower() == 'female':
        prediction_input = pd.DataFrame([[0]], columns=['Gender_Male'])
    else:
        return jsonify({'error': 'Invalid gender. Please use Male or Female.'}), 400

    predicted_salary = model.predict(prediction_input)[0]
    return jsonify({'predicted_salary': predicted_salary})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
