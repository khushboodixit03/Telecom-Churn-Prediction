from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and scaler
model = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define categorical columns
categorical_cols = ['gender','SeniorCitizen','Partner','Dependents',
                    'PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies',
                    'Contract','PaperlessBilling','PaymentMethod']

# Load dummy data to fit encoders
dummy_data = pd.read_csv('dummy.csv')
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    dummy_data[col] = dummy_data[col].astype(str)
    encoders[col] = le.fit(dummy_data[col])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            form_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'Tenure': float(request.form['Tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges'])
            }

            # Convert to DataFrame
            df = pd.DataFrame([form_data])

            # Encode categorical features
            for col in categorical_cols:
                df[col] = encoders[col].transform(df[col].astype(str))

            # Scale numerical features
            df[['Tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
                df[['Tenure', 'MonthlyCharges', 'TotalCharges']]
            )

            # Predict
            prob = model.predict_proba(df)[:, 1][0]
            prediction = f"{prob * 100:.2f}%"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
