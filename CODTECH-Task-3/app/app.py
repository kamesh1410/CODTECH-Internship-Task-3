from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib
import os

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form data to model-compatible format
        input_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'membership': request.form['membership'],
            'tenure': int(request.form['tenure']),
            'monthly_charge': float(request.form['monthly_charge']),
            'total_spend': float(request.form['total_spend']),
            'login_frequency': int(request.form['login_frequency']),
            'support_tickets': int(request.form['support_tickets']),
            'payment_delays': int(request.form['payment_delays']),
            'feature_usage': int(request.form['feature_usage']),
            'last_login': int(request.form['last_login'])
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        return render_template('result.html',
                            prediction='Churn' if prediction == 1 else 'No Churn',
                            probability=f"{probability*100:.1f}%",
                            input_data=input_data)
    
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)