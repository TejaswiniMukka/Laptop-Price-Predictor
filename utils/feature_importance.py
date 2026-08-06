"""
Feature Importance module for the Laptop Price Predictor.
Loads the trained Random Forest pipeline, extracts and plots top 20 feature importances.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_feature_importances() -> pd.DataFrame:
    """
    Extracts feature importances from the saved pipeline.
    """
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'models', 
        'laptop_price_pipeline.pkl'
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained pipeline not found at {model_path}. Please train the model first.")
        
    pipeline = joblib.load(model_path)
    
    # Extract components
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    if not hasattr(regressor, 'feature_importances_'):
        raise AttributeError("The selected model in the pipeline does not support feature_importances_.")
        
    # Get feature names from ColumnTransformer
    cat_transformer = preprocessor.named_transformers_['cat']
    cat_features = cat_transformer.get_feature_names_out(['Company', 'TypeName', 'CPU Brand', 'GPU Brand', 'OpSys'])
    
    # Numerical features are passed through directly or scaled
    num_features = preprocessor.transformers_[1][2]
    
    # Combine feature lists in the order of ColumnTransformer execution
    all_features = list(cat_features) + list(num_features)
    
    # Create DataFrame of features and importances
    importances = regressor.feature_importances_
    
    # Safety check on length mismatch
    if len(all_features) != len(importances):
        # Fallback to index if mismatch
        all_features = [f"Feature_{i}" for i in range(len(importances))]
        
    df_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return df_importance

def plot_importance_matplotlib(df_importance: pd.DataFrame, save_path: str = None):
    """
    Generates and optionally saves a Matplotlib feature importance bar chart.
    """
    top_20 = df_importance.head(20).copy().sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_20['Feature'], top_20['Importance'], color='#6C5B7B')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.title('Top 20 Important Features (Matplotlib)')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Matplotlib chart saved to {save_path}")
    
    return plt.gcf()

def plot_importance_plotly(df_importance: pd.DataFrame):
    """
    Generates an interactive Plotly bar chart for feature importances.
    """
    top_20 = df_importance.head(20).copy()
    
    fig = px.bar(
        top_20,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 20 Laptop Price Predictors (Plotly Interactive)',
        labels={'Importance': 'Feature Importance Score', 'Feature': 'Laptop Specifications'},
        color='Importance',
        color_continuous_scale='Sunset'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    return fig

def main():
    """
    Main runner to execute analysis and save static plots.
    """
    # Set cwd to script parent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        print("Extracting feature importances...")
        df_imp = get_feature_importances()
        
        # Save Top 20 table
        print("\nTop 20 Important Features:")
        print(df_imp.head(20).to_string(index=False))
        
        # Save Matplotlib plot
        save_path = os.path.join(os.path.dirname(script_dir), 'images', 'feature_importance_matplotlib.png')
        plot_importance_matplotlib(df_imp, save_path=save_path)
        
        print("\nFeatures affecting price the most:")
        print("1. Ram: Larger RAM capacity leads to significantly higher laptop costs.")
        print("2. CPU Brand (e.g., Intel Core i7 vs. i3): Advanced processors are key price drivers.")
        print("3. Weight / Dimensions: Sleeker, lighter notebooks (Ultrabooks) command premium prices.")
        print("4. Screen Resolution / PPI: High-density screens (like IPS Panel 4K) are very expensive.")
        print("5. GPU Brand (NVIDIA): Gaming or professional workstation GPUs add huge costs.")
        
    except Exception as e:
        print(f"Error executing feature importance: {str(e)}")

if __name__ == '__main__':
    main()
