"""
Titanic Survival Prediction Web Application
Flask app to serve the trained model via a web interface
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Global variables to store model, scaler, and encoders
model = None
scaler = None
label_encoders = None
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']


def load_model():
    """Load the trained model, scaler, and encoders from disk."""
    global model, scaler, label_encoders
    
    try:
        # Load model
        with open('model/titanic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open('model/titanic_model_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open('model/titanic_model_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        print("Model, scaler, and encoders loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        return False


@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the web interface.
    
    Expected JSON format:
    {
        "Pclass": int,
        "Sex": str (male/female),
        "Age": float,
        "SibSp": int,
        "Fare": float
    }
    
    Returns:
    {
        "success": bool,
        "prediction": int (0 or 1),
        "prediction_label": str,
        "probability_not_survived": float,
        "probability_survived": float,
        "confidence": float
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Missing required fields. Please provide: Pclass, Sex, Age, SibSp, Fare'
            }), 400
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([{
            'Pclass': int(data['Pclass']),
            'Sex': str(data['Sex']).lower(),
            'Age': float(data['Age']),
            'SibSp': int(data['SibSp']),
            'Fare': float(data['Fare'])
        }])
        
        # Encode Sex
        input_data['Sex'] = label_encoders['Sex'].transform(input_data['Sex'])
        
        # Select features
        X = input_data[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'SURVIVED' if prediction == 1 else 'DID NOT SURVIVE',
            'probability_not_survived': float(probabilities[0]),
            'probability_survived': float(probabilities[1]),
            'confidence': float(max(probabilities))
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input data: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify model is loaded."""
    if model is None or scaler is None or label_encoders is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'message': 'Model is ready for predictions'
    })


if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        print("Starting Titanic Survival Prediction Web Application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure the model files exist.")
        print("Expected files: titanic_model.pkl, titanic_model_scaler.pkl, titanic_model_encoders.pkl")