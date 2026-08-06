"""
Encoding module for the Laptop Price Predictor.
Sets up the ColumnTransformer and OneHotEncoder for preprocessing categorical columns.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor(categorical_cols=None, numerical_cols=None) -> ColumnTransformer:
    """
    Creates and returns a ColumnTransformer for preprocessing.
    - OneHotEncoder for categorical features.
    - StandardScaler for numerical features (optional, depending on the model, but good practice).
    """
    if categorical_cols is None:
        categorical_cols = ['Company', 'TypeName', 'CPU Brand', 'GPU Brand', 'OpSys']
        
    if numerical_cols is None:
        numerical_cols = [
            'Inches', 'Ram', 'Weight', 'Touchscreen', 'IPS Panel', 
            'Resolution Width', 'Resolution Height', 'PPI', 
            'CPU Clock Speed', 'SSD', 'HDD', 'Flash Storage', 'Hybrid', 'Total Storage'
        ]
        
    # Configure OneHotEncoder to handle unknown categories during inference gracefully
    categorical_transformer = OneHotEncoder(
        sparse_output=False, 
        handle_unknown='ignore'
    )
    
    # We can pass through or scale the numerical features. Let's scale them for models like Linear Regression
    # and keep them scaled for trees (does not affect tree splits, but standardizes the features)
    # Alternatively, we can use remainder='passthrough'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='drop' # Drops any columns not explicitly handled (like target or others)
    )
    
    return preprocessor
