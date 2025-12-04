
from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model/calories_model.pkl')
encoders = joblib.load('model/label_encoders.pkl')

# Load dataset for dynamic activity options
df = pd.read_csv('model/calories_burned_dataset.csv')
activities = df['Activity'].unique().tolist()

@app.route('/')
def index():
    return render_template('index.html', activities=activities)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        age = float(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        activity = request.form['activity']

        # Input validation
        if age <= 0 or height <= 0 or weight <= 0 or duration <= 0 or heart_rate <= 0 or body_temp <= 0:
            return "Error: All numerical values must be positive."
        
        if not (30 <= heart_rate <= 200):
            return "Error: Heart rate must be between 30 and 200."
        if not (95 <= body_temp <= 105):
            return "Error: Body temperate must be between 95 °F and 105 °F."

        # Encode categorical
        gender_enc = encoders['gender'].transform([gender])[0]
        activity_enc = encoders['activity'].transform([activity])[0]

        input_data = np.array([[age, gender_enc, height, weight, duration, heart_rate, body_temp, activity_enc]])
        calories_burned = model.predict(input_data)[0]

        # Save prediction to CSV
        csv_file = os.path.join(BASE_DIR, 'model', 'user_predictions.csv')
        new_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Heart_Rate': heart_rate,
            'Body_Temp': body_temp,
            'Activity': activity,
            'Predicted_Calories': round(calories_burned, 2)
        }])

        if os.path.exists(csv_file):
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            new_data.to_csv(csv_file, index=False)

        # Prepare chart data for durations
        durations = list(range(5, 61, 5))
        calories_values = []
        for d in durations:
            temp_input = np.array([[age, gender_enc, height, weight, d, heart_rate, body_temp, activity_enc]])
            calories_values.append(model.predict(temp_input)[0])

        return render_template(
            'result.html',
            calories=round(calories_burned, 2),
            durations=durations,
            calories_values=calories_values
        )
    except Exception as e:
        return f"Error: {e}"

@app.route('/download')
def download():
    csv_file = os.path.join(BASE_DIR, 'model', 'user_predictions.csv')
    if os.path.exists(csv_file):
        # Send the file to the user for download
        return send_file(csv_file, as_attachment=True)
    else:
        return "No predictions available yet. Please make a prediction first."

if __name__ == '__main__':
    app.run(debug=False, host= "0.0.0.0", port=5000)
