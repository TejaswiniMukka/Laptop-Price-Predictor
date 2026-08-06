"""
Prediction module for the Laptop Price Predictor.
Loads the trained pipeline and runs inference on user input.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure root project path is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'models', 
    'laptop_price_pipeline.pkl'
)

def load_prediction_pipeline():
    """
    Loads the saved joblib ML pipeline.
    """
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Model pipeline not found at {PIPELINE_PATH}. Please run training first.")
    
    # Load pipeline
    pipeline = joblib.load(PIPELINE_PATH)
    return pipeline

def predict_price(input_data: dict) -> tuple:
    """
    Predicts the price of a laptop given its specifications.
    
    Input:
    input_data (dict): Dictionary with keys:
      - Company (str)
      - Product (str)
      - TypeName (str)
      - Inches (float)
      - ScreenResolution (str)
      - Cpu (str)
      - Ram (str) e.g., "8GB"
      - Memory (str) e.g., "256GB SSD"
      - Gpu (str)
      - OpSys (str)
      - Weight (str) e.g., "1.37kg"
      
    Returns:
    tuple: (predicted_price: float, lower_bound: float, upper_bound: float)
    """
    try:
        # Load the pipeline
        pipeline = load_prediction_pipeline()
        
        # Convert single input dict to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Predict using pipeline (automatically applies preprocess_pipeline and ColumnTransformer)
        predicted_price = float(pipeline.predict(df_input)[0])
        
        # Estimate range (e.g. +/- 10% as estimated range boundary)
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10
        
        return predicted_price, lower_bound, upper_bound
        
    except Exception as e:
        # Return error description and empty range
        raise RuntimeError(f"Error during price prediction: {str(e)}")
