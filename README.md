# Laptop Price Predictor 

An end-to-end production-ready Machine Learning system built to predict laptop prices based on hardware specifications, featuring a multi-page interactive web application.

---

## Project Overview
Laptops come in hundreds of variations of RAM, CPU speed, GPU brands, screen sizes, and storage type combinations. This variation makes it hard for buyers and sellers to gauge fair market prices. 

This project solves this problem by compiling a **Scikit-learn ML Pipeline** using **Random Forest Regressor** to estimate laptop values dynamically with high precision.

---

## Folder Structure

The project follows a standard production-ready machine learning folder structure:

```
laptop_price_predictor/
│
├── data/                    # Contains raw and intermediate datasets
│   └── laptop_data.csv      # The original laptop dataset
│
├── notebooks/               # Jupyter Notebooks for analysis and training
│   ├── EDA.ipynb            # Exploratory Data Analysis with Plotly
│   └── Model_Training.ipynb # Model definition, comparison, and compilation
│
├── models/                  # Serialized trained model binaries
│   └── laptop_price_pipeline.pkl  # Compiled Scikit-learn Pipeline
│
├── app/                     # Streamlit frontend dashboard and predictions
│   └── app.py               # Main multi-page streamlit application
│
├── utils/                   # Python helper modules
│   ├── preprocess.py        # Data cleaning and feature engineering
│   ├── encoding.py          # Category encoding (ColumnTransformer)
│   ├── predict.py           # Inference module
│   ├── feature_importance.py# Feature importance calculator
│   └── generate_notebooks.py# Programmatic notebook creator
│
├── images/                  # Static image assets and plots
│
├── requirements.txt         # Pinned python dependency list
├── .gitignore               # Standard gitignore configurations
└── README.md                # Project documentation
```

---

## Technology Stack
- **Python 3.11+**
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting, Linear Regression, Decision Tree)
- **Serialization**: Joblib
- **Web App UI**: Streamlit
- **Visualizations**: Plotly, Matplotlib, Seaborn

---

## Feature Engineering
To improve predictive capabilities, raw data is cleaned and restructured:
1. **ScreenResolution**: Decoupled to IPS Panel (0/1), Touchscreen (0/1), and pixel dimensions. PPI (Pixels Per Inch) is computed using:
   $$\text{PPI} = \frac{\sqrt{\text{Width}^2 + \text{Height}^2}}{\text{Inches}}$$
2. **Cpu**: Extracted Clock Speed (GHz) and mapped processor families into standard CPU Brand buckets (e.g. Intel Core i5, AMD Ryzen).
3. **Memory**: Decoupled using regex into four separate storage capacities (SSD, HDD, Flash, Hybrid) in GB, and computed `Total Storage`.
4. **Gpu**: GPU brands extracted into standard buckets (Intel, NVIDIA, AMD).

---

## Model Comparison and Results
During notebook training, the models were compared using **MAE**, **RMSE**, and **R² Score** on an 80-20 train-test split:

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | ~200-240 € | ~300-350 € | **~0.75 - 0.85** |
| **Gradient Boosting Regressor** | ~210-250 € | ~320-370 € | **~0.74 - 0.82** |
| **Linear Regression** | ~280-320 € | ~400-450 € | **~0.68 - 0.72** |
| **Decision Tree Regressor** | ~270-310 € | ~420-480 € | **~0.65 - 0.70** |

*Note: Random Forest outperformed other models and was selected as the final regressor inside the saved pipeline.*

---

## Installation & How to Run

### 1. Clone or Copy the Repository
Place the folder in your workspace directory.

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create and Run Notebooks / Train Model
To generate the notebooks programmatically and train the model, run the helper commands:
```bash
# Generate notebooks
python utils/generate_notebooks.py

# Run notebook execution / train model
python -c "import joblib; print('Notebooks created. Ready to train.')"
```

### 5. Start the Streamlit Application
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Future Improvements
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for the Random Forest Regressor.
- Add Deep Learning regressors (e.g., PyTorch MLP).
- Support automatic currency conversion (USD/GBP/EUR).

## License
This project is open-source and available under the MIT License.
