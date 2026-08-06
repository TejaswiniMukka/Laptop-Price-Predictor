"""
Helper script to execute the generated notebooks (EDA.ipynb and Model_Training.ipynb) programmatically.
This populates the cell outputs, creates the model pipeline binary, and saves the executed notebook states.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    print(f"Executing notebook: {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # Create the execution preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Execute the notebook cells
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    # Save the executed notebook back to disk
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Finished executing {notebook_path}.")

def main():
    # Set directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    eda_path = os.path.join(os.path.dirname(script_dir), 'notebooks', 'EDA.ipynb')
    train_path = os.path.join(os.path.dirname(script_dir), 'notebooks', 'Model_Training.ipynb')
    
    try:
        # Run EDA first
        run_notebook(eda_path)
        
        # Run Model Training second (this trains the model and generates models/laptop_price_pipeline.pkl)
        run_notebook(train_path)
        
        print("\nAll notebooks executed successfully. Model pipeline binary created.")
        
    except Exception as e:
        print(f"Error executing notebooks: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
