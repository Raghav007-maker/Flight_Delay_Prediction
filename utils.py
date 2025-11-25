import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime

def check_project_files():
    required_files = {
        'data/flights_processed.csv': 'Processed flight data',
        'models/flight_delay_model.pkl': 'Trained ML model',
        'models/airline_encoder.pkl': 'Airline encoder',
        'models/origin_encoder.pkl': 'Origin encoder',
        'models/feature_names.pkl': 'Feature names'
    }
    
    missing = []
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing.append(f"{description} ({file})")
    
    return len(missing) == 0, missing

def safe_load_models():
    try:
        with open('models/flight_delay_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/airline_encoder.pkl', 'rb') as f:
            airline_encoder = pickle.load(f)
        
        with open('models/origin_encoder.pkl', 'rb') as f:
            origin_encoder = pickle.load(f)
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, airline_encoder, origin_encoder, feature_names, None
    
    except FileNotFoundError as e:
        return None, None, None, None, f"File not found: {e}"
    except Exception as e:
        return None, None, None, None, f"Error loading models: {e}"

def validate_api_key(api_key):
    if not api_key:
        return False, "API key is empty"
    if len(api_key) < 20:
        return False, "API key seems too short"
    return True, "Valid"

def format_probability(prob):
    return f"{prob*100:.1f}%"

def get_risk_level(probability):
    if probability < 0.3:
        return '🟢 LOW', 'green', 'success'
    elif probability < 0.6:
        return '🟡 MEDIUM', 'orange', 'warning'
    else:
        return '🔴 HIGH', 'red', 'error'

def log_prediction(flight_data, prediction, filename='prediction_log.csv'):
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'flight': flight_data.get('flight', 'N/A'),
            'airline': flight_data.get('airline', 'N/A'),
            'route': flight_data.get('route', 'N/A'),
            'probability': prediction.get('probability', 0),
            'prediction': prediction.get('prediction', 'N/A'),
            'risk_level': prediction.get('risk_level', 'N/A')
        }
        
        df_log = pd.DataFrame([log_entry])
        
        if os.path.exists(filename):
            df_log.to_csv(filename, mode='a', header=False, index=False)
        else:
            df_log.to_csv(filename, index=False)
        
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False