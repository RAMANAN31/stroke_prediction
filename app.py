"""
from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
with open('stroke_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extracting the form data
    try:
        age = int(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        avg_glucose_level = float(data['avg_glucose_level'])
        bmi = float(data['bmi'])
    except ValueError:
        return "Invalid input! Please enter numeric values for age, hypertension, heart disease, glucose level, and BMI."

    # Convert categorical inputs into one-hot encoded format
    gender_male = 1 if data['gender'] == 'Male' else 0
    ever_married_yes = 1 if data['ever_married'] == 'Yes' else 0

    # One-hot encoding for work type
    work_type_private = 1 if data['work_type'] == 'Private' else 0
    work_type_self_employed = 1 if data['work_type'] == 'Self-employed' else 0
    work_type_children = 1 if data['work_type'] == 'Children' else 0
    work_type_govt_job = 1 if data['work_type'] == 'Govt_job' else 0
    work_type_never_worked = 1 if data['work_type'] == 'Never_worked' else 0  # Added this line

    # One-hot encoding for residence type
    residence_type_urban = 1 if data['Residence_type'] == 'Urban' else 0

    # One-hot encoding for smoking status
    smoking_status_smokes = 1 if data['smoking_status'] == 'smokes' else 0
    smoking_status_never_smoked = 1 if data['smoking_status'] == 'never smoked' else 0
    smoking_status_formerly_smoked = 1 if data['smoking_status'] == 'formerly smoked' else 0

    # Prepare input array (ensure all 16 features are included)
    input_data = [
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender_male, ever_married_yes,
        work_type_private, work_type_self_employed, work_type_children, work_type_govt_job, work_type_never_worked,
        residence_type_urban,
        smoking_status_smokes, smoking_status_never_smoked, smoking_status_formerly_smoked
    ]

    # Convert input to numpy array and reshape for prediction
    input_array = np.array(input_data).reshape(1, -1)

    # Check if input data has the same number of features as the model expects
    expected_features = model.n_features_in_
    if input_array.shape[1] != expected_features:
        return f"Feature mismatch! Model expects {expected_features} features, but got {input_array.shape[1]}."

    # Make prediction
    prediction = model.predict(input_array)[0]

    # Interpret the prediction
    result = "Stroke" if prediction == 1 else "No Stroke"
    return f"Prediction: {result}"

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
with open('stroke_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract and process input data
    try:
        age = int(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        avg_glucose_level = float(data['avg_glucose_level'])
        bmi = float(data['bmi'])
    except ValueError:
        return "Invalid input! Please enter numeric values for age, hypertension, heart disease, glucose level, and BMI."

    # Convert categorical inputs into one-hot encoded format
    gender_male = 1 if data['gender'] == 'Male' else 0
    ever_married_yes = 1 if data['ever_married'] == 'Yes' else 0

    work_type_private = 1 if data['work_type'] == 'Private' else 0
    work_type_self_employed = 1 if data['work_type'] == 'Self-employed' else 0
    work_type_children = 1 if data['work_type'] == 'Children' else 0
    work_type_govt_job = 1 if data['work_type'] == 'Govt_job' else 0
    work_type_never_worked = 1 if data['work_type'] == 'Never_worked' else 0

    residence_type_urban = 1 if data['Residence_type'] == 'Urban' else 0

    smoking_status_smokes = 1 if data['smoking_status'] == 'smokes' else 0
    smoking_status_never_smoked = 1 if data['smoking_status'] == 'never smoked' else 0
    smoking_status_formerly_smoked = 1 if data['smoking_status'] == 'formerly smoked' else 0

    input_data = [
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender_male, ever_married_yes,
        work_type_private, work_type_self_employed, work_type_children, work_type_govt_job, work_type_never_worked,
        residence_type_urban,
        smoking_status_smokes, smoking_status_never_smoked, smoking_status_formerly_smoked
    ]

    input_array = np.array(input_data).reshape(1, -1)

    expected_features = model.n_features_in_
    if input_array.shape[1] != expected_features:
        return f"Feature mismatch! Model expects {expected_features} features, but got {input_array.shape[1]}."

    prediction = model.predict(input_array)[0]
    result = "Stroke" if prediction == 1 else "No Stroke"

    # Suggestions to prevent stroke
    suggestions = [
        "Maintain a healthy diet rich in fruits and vegetables.",
        "Exercise regularly, aiming for at least 30 minutes a day.",
        "Avoid smoking and limit alcohol consumption.",
        "Monitor and manage blood pressure, cholesterol, and diabetes.",
        "Maintain a healthy weight and manage stress effectively."
    ]

    return render_template('result.html', prediction=result, suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)
