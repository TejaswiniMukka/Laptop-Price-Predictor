"""
Preprocessing module for the Laptop Price Predictor.
Provides cleaning and feature engineering functions for training and inference.
"""

import re
import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw laptop price data:
    - Removes duplicate rows.
    - Removes laptop_ID.
    - Handles missing values.
    - Standardizes RAM and Weight.
    """
    # Create a copy to prevent SettingWithCopyWarning
    df = df.copy()
    
    # 1. Remove duplicate rows
    df = df.drop_duplicates()
    
    # 2. Remove laptop_ID if present
    if 'laptop_ID' in df.columns:
        df = df.drop(columns=['laptop_ID'])
        
    # 3. Handle missing values (simple dropna for training, fillna could be handled inside pipeline)
    df = df.dropna()
    
    # 4. Convert Ram: "8GB" -> 8 (int)
    if 'Ram' in df.columns and not pd.api.types.is_numeric_dtype(df['Ram']):
        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).astype(int)
        
    # 5. Convert Weight: "1.37kg" -> 1.37 (float)
    if 'Weight' in df.columns and not pd.api.types.is_numeric_dtype(df['Weight']):
        df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', regex=False).astype(float)
        
    return df

def extract_screen_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from ScreenResolution:
    - Touchscreen (0/1)
    - IPS Panel (0/1)
    - Resolution Width (int)
    - Resolution Height (int)
    - PPI (Pixels Per Inch) (float)
    """
    df = df.copy()
    if 'ScreenResolution' in df.columns:
        # Extract Touchscreen presence
        df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        
        # Extract IPS Panel presence
        df['IPS Panel'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS Panel' in x else 0)
        
        # Extract resolution width and height (e.g. "1920x1080")
        def parse_resolution(resolution_str):
            matches = re.findall(r'(\d+)x(\d+)', resolution_str)
            if matches:
                return int(matches[0][0]), int(matches[0][1])
            return 1920, 1080 # Fallback default
            
        resolutions = df['ScreenResolution'].apply(parse_resolution)
        df['Resolution Width'] = [res[0] for res in resolutions]
        df['Resolution Height'] = [res[1] for res in resolutions]
        
        # Calculate PPI = sqrt(width^2 + height^2) / inches
        # Fallback to avoid division by zero
        df['PPI'] = np.sqrt(df['Resolution Width']**2 + df['Resolution Height']**2) / df['Inches'].replace(0, 1)
        
    return df

def extract_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from Cpu:
    - CPU Brand (Intel Core i7, Intel Core i5, Intel Core i3, AMD Ryzen, Intel Celeron, etc.)
    - CPU Clock Speed (GHz) (float)
    """
    df = df.copy()
    if 'Cpu' in df.columns:
        # Extract Clock Speed in GHz
        def get_clock_speed(cpu_str):
            match = re.search(r'(\d+\.?\d*)\s*GHz', cpu_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            return 2.0 # Default fallback clock speed
            
        df['CPU Clock Speed'] = df['Cpu'].apply(get_clock_speed)
        
        # Extract CPU Brand category
        def get_cpu_brand(cpu_str):
            cpu_name = ' '.join(cpu_str.split()[:-1]) # Remove the GHz part
            if 'Intel Core i7' in cpu_name:
                return 'Intel Core i7'
            elif 'Intel Core i5' in cpu_name:
                return 'Intel Core i5'
            elif 'Intel Core i3' in cpu_name:
                return 'Intel Core i3'
            elif 'Intel Celeron' in cpu_name:
                return 'Intel Celeron'
            elif 'Intel Pentium' in cpu_name:
                return 'Intel Pentium'
            elif 'AMD Ryzen' in cpu_name:
                return 'AMD Ryzen'
            elif 'AMD A-Series' in cpu_name or 'AMD A9' in cpu_name or 'AMD A6' in cpu_name or 'AMD A10' in cpu_name or 'AMD E-Series' in cpu_name or 'AMD FX' in cpu_name:
                return 'AMD Processor'
            elif 'Intel' in cpu_name:
                return 'Other Intel Processor'
            else:
                return 'Other Processor'
                
        df['CPU Brand'] = df['Cpu'].apply(get_cpu_brand)
        
    return df

def extract_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts storage capacity features from Memory column:
    - SSD Capacity (GB)
    - HDD Capacity (GB)
    - Flash Storage Capacity (GB)
    - Hybrid Storage Capacity (GB)
    - Total Storage Capacity (GB)
    """
    df = df.copy()
    if 'Memory' in df.columns:
        # Standardize representation
        df['Memory'] = df['Memory'].astype(str).str.replace(r'\.0', '', regex=True)
        df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
        df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False) # Convert TB to GB
        
        # Extract capacity functions
        def parse_storage(memory_str, storage_type):
            # E.g. "128 SSD + 1000 HDD"
            parts = memory_str.split('+')
            capacity = 0
            for part in parts:
                if storage_type in part:
                    # Find all digits in this part
                    nums = re.findall(r'\d+', part)
                    if nums:
                        capacity += int(nums[0])
            return capacity
            
        df['SSD'] = df['Memory'].apply(lambda x: parse_storage(x, 'SSD'))
        df['HDD'] = df['Memory'].apply(lambda x: parse_storage(x, 'HDD'))
        df['Flash Storage'] = df['Memory'].apply(lambda x: parse_storage(x, 'Flash Storage'))
        df['Hybrid'] = df['Memory'].apply(lambda x: parse_storage(x, 'Hybrid'))
        
        # Calculate Total Storage
        df['Total Storage'] = df['SSD'] + df['HDD'] + df['Flash Storage'] + df['Hybrid']
        
    return df

def extract_gpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts GPU Brand from Gpu:
    - GPU Brand (Intel, AMD, NVIDIA, etc.)
    """
    df = df.copy()
    if 'Gpu' in df.columns:
        df['GPU Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
        
    return df

def preprocess_pipeline(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Runs the complete preprocessing pipeline.
    If is_training is True, expects target 'Price_euros' and filters it.
    If is_training is False (inference), can predict without 'Price_euros'.
    """
    # 1. Clean basic columns and values
    df_clean = clean_data(df)
    
    # 2. Extract engineered features
    df_features = extract_screen_features(df_clean)
    df_features = extract_cpu_features(df_features)
    df_features = extract_memory_features(df_features)
    df_features = extract_gpu_features(df_features)
    
    # 3. Drop unnecessary original columns and columns with extremely high cardinality (e.g. Product)
    cols_to_drop = ['ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'Product']
    # If laptop_ID exists, drop it
    if 'laptop_ID' in df_features.columns:
        cols_to_drop.append('laptop_ID')
        
    df_final = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])
    
    return df_final

def raw_preprocess_transformer(df):
    """
    Helper function to clean and engineer features inside a Scikit-learn Pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return preprocess_pipeline(df, is_training=False)
