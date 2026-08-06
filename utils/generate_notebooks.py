"""
Helper script to generate the project Jupyter Notebooks (EDA.ipynb and Model_Training.ipynb) programmatically.
This ensures they have correct cells, formatting, and markdown explanations.
"""

import json
import os

def create_eda_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Laptop Price Predictor - Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "This notebook performs an in-depth exploratory analysis of the laptop price dataset to understand the relationships, distributions, and patterns in the data."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import plotly.express as px\n",
                    "import plotly.io as pio\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import sys\n",
                    "import os\n",
                    "\n",
                    "# Set plotly template to professional dark/light theme\n",
                    "pio.templates.default = \"plotly_white\"\n",
                    "\n",
                    "# Load dataset\n",
                    "df = pd.read_csv('../data/laptop_data.csv', encoding='latin-1')\n",
                    "print(f\"Dataset shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 1: Dataset Overview\n",
                    "We inspect the column data types and check for any immediate format issues."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.info()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Missing Value Analysis\n",
                    "We count the number of missing values in each column to identify if imputation is required."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Missing values:\")\n",
                    "print(df.isnull().sum())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Duplicate Detection\n",
                    "We check if there are any duplicate records in the dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Number of duplicate rows: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Statistical Summary\n",
                    "We print descriptive statistics for numerical and categorical columns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.describe(include='all')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 2: Target Variable (Price_euros) Distribution\n",
                    "We visualize the distribution of laptop prices. Typically, price distributions are right-skewed, which might suggest a log-transformation during modeling (or robust models like Random Forest)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_price = px.histogram(df, x='Price_euros', marginal='box', \n",
                    "                         title='Distribution of Laptop Prices (in Euros)',\n",
                    "                         color_discrete_sequence=['#636EFA'])\n",
                    "fig_price.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 3: Laptop Brand (Company) Distribution\n",
                    "We examine which manufacturers dominate the dataset. This helps us see if we have enough sample sizes for all brands."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "brand_counts = df['Company'].value_counts().reset_index()\n",
                    "brand_counts.columns = ['Company', 'Count']\n",
                    "fig_brand = px.bar(brand_counts, x='Company', y='Count', \n",
                    "                    title='Laptop Brand Distribution',\n",
                    "                    color='Count', color_continuous_scale='Purples')\n",
                    "fig_brand.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 4: Laptop Type (TypeName) Distribution\n",
                    "We inspect the counts of laptop types (e.g. Notebook, Ultrabook, Gaming, 2 in 1 Convertible) to see consumer categories."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "type_counts = df['TypeName'].value_counts().reset_index()\n",
                    "type_counts.columns = ['TypeName', 'Count']\n",
                    "fig_type = px.pie(type_counts, names='TypeName', values='Count', \n",
                    "                    title='Laptop Type Distribution',\n",
                    "                    color_discrete_sequence=px.colors.qualitative.Pastel)\n",
                    "fig_type.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 5: Operating System Distribution\n",
                    "We look at the spread of operating systems."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "opsys_counts = df['OpSys'].value_counts().reset_index()\n",
                    "opsys_counts.columns = ['OpSys', 'Count']\n",
                    "fig_opsys = px.bar(opsys_counts, x='OpSys', y='Count', \n",
                    "                    title='Operating System Distribution',\n",
                    "                    color='Count', color_continuous_scale='Viridis')\n",
                    "fig_opsys.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 6: RAM Distribution\n",
                    "Before analyzing RAM, we clean it (extracting number from \"8GB\")."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_clean = df.copy()\n",
                    "df_clean['Ram_GB'] = df_clean['Ram'].str.replace('GB', '', regex=False).astype(int)\n",
                    "ram_counts = df_clean['Ram_GB'].value_counts().sort_index().reset_index()\n",
                    "ram_counts.columns = ['Ram_GB', 'Count']\n",
                    "fig_ram = px.bar(ram_counts, x='Ram_GB', y='Count', \n",
                    "                  title='RAM Size Distribution (GB)',\n",
                    "                  color='Count', color_continuous_scale='Cividis')\n",
                    "fig_ram.update_layout(xaxis=dict(type='category'))\n",
                    "fig_ram.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 7: Price Distribution by Company\n",
                    "A boxplot showing price variations by brand. This shows that brands like Apple, Dell, and MSI target higher price ranges while Acer targets budget markets."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_price_brand = px.box(df, x='Company', y='Price_euros', \n",
                    "                         title='Price Distribution by Company', color='Company')\n",
                    "fig_price_brand.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Correlation Analysis & Outlier Detection\n",
                    "We clean the data using our modular preprocessing pipeline to convert textual features to numeric values, enabling correlation analysis."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sys.path.append('../')\n",
                    "from utils.preprocess import preprocess_pipeline\n",
                    "\n",
                    "df_preprocessed = preprocess_pipeline(df, is_training=True)\n",
                    "\n",
                    "# Select numerical columns\n",
                    "num_cols = df_preprocessed.select_dtypes(include=[np.number])\n",
                    "corr_matrix = num_cols.corr()\n",
                    "\n",
                    "fig_corr = px.imshow(corr_matrix, text_auto=True, \n",
                    "                      title='Correlation Matrix of Features',\n",
                    "                      color_continuous_scale='RdBu_r', aspect='auto')\n",
                    "fig_corr.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Chart 8: Price vs. PPI (Pixels Per Inch)\n",
                    "We investigate how screen density relates to pricing."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_ppi = px.scatter(df_preprocessed, x='PPI', y='Price_euros', \n",
                    "                     title='Price vs. Pixels Per Inch (PPI)',\n",
                    "                     color='TypeName', hover_name='Company')\n",
                    "fig_ppi.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs('../notebooks', exist_ok=True)
    with open('../notebooks/EDA.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("EDA.ipynb generated.")

def create_model_training_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Laptop Price Predictor - Model Training\n",
                    "\n",
                    "In this notebook, we load the raw dataset, perform model training and validation, and save the final pipeline using joblib."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import sys\n",
                    "import os\n",
                    "import joblib\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.preprocessing import FunctionTransformer\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "\n",
                    "# Model definitions\n",
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.tree import DecisionTreeRegressor\n",
                    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                    "\n",
                    "# Project imports\n",
                    "sys.path.append('../')\n",
                    "from utils.preprocess import preprocess_pipeline, raw_preprocess_transformer\n",
                    "from utils.encoding import get_preprocessor\n",
                    "\n",
                    "# Load raw dataset\n",
                    "raw_df = pd.read_csv('../data/laptop_data.csv', encoding='latin-1')\n",
                    "print(f\"Loaded raw data shape: {raw_df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Data Splitting (80% Train, 20% Test)\n",
                    "We split the RAW dataset before training to prevent any data leakage. Preprocessing will be done inside the training loop / evaluation pipeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X = raw_df.drop(columns=['Price_euros'])\n",
                    "y = raw_df['Price_euros']\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "print(f\"Train size: {X_train.shape}, Test size: {X_test.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Defining the Processing Pipeline\n",
                    "We wrap the data cleaning and feature engineering step inside a `FunctionTransformer` so that it fits seamlessly into scikit-learn's Pipeline, allowing direct prediction on raw data!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test the transformer on training sample to extract column configurations\n",
                    "X_train_processed = raw_preprocess_transformer(X_train)\n",
                    "cat_cols = ['Company', 'TypeName', 'CPU Brand', 'GPU Brand', 'OpSys']\n",
                    "num_cols = [col for col in X_train_processed.columns if col not in cat_cols]\n",
                    "\n",
                    "# Get encoding preprocessor\n",
                    "preprocessor = get_preprocessor(categorical_cols=cat_cols, numerical_cols=num_cols)\n",
                    "print(\"Preprocessor and Transformer configured.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Train and Compare Models\n",
                    "We train and compare four regressor algorithms: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = {\n",
                    "    'Linear Regression': LinearRegression(),\n",
                    "    'Decision Tree': DecisionTreeRegressor(random_state=42),\n",
                    "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),\n",
                    "    'Gradient Boosting': GradientBoostingRegressor(random_state=42)\n",
                    "}\n",
                    "\n",
                    "results = []\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    # Create complete Pipeline\n",
                    "    pipe = Pipeline([\n",
                    "        ('raw_preprocess', FunctionTransformer(raw_preprocess_transformer)),\n",
                    "        ('preprocessor', preprocessor),\n",
                    "        ('regressor', model)\n",
                    "    ])\n",
                    "    \n",
                    "    # Train pipeline directly on raw data!\n",
                    "    pipe.fit(X_train, y_train)\n",
                    "    \n",
                    "    # Predict on raw data!\n",
                    "    y_pred = pipe.predict(X_test)\n",
                    "    \n",
                    "    # Metrics calculation\n",
                    "    mae = mean_absolute_error(y_test, y_pred)\n",
                    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
                    "    r2 = r2_score(y_test, y_pred)\n",
                    "    \n",
                    "    results.append({\n",
                    "        'Model': name,\n",
                    "        'MAE': mae,\n",
                    "        'RMSE': rmse,\n",
                    "        'R² Score': r2\n",
                    "    })\n",
                    "\n",
                    "results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)\n",
                    "results_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Select the Best Model\n",
                    "We select the model with the highest R² score."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best_model_name = results_df.iloc[0]['Model']\n",
                    "print(f\"Best model is {best_model_name} with R² score of {results_df.iloc[0]['R² Score']:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Save the Best Pipeline\n",
                    "We build the final pipeline with the best performing regressor (e.g. Random Forest or Gradient Boosting) and save it under `models/laptop_price_pipeline.pkl` using joblib."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best_model_instance = models[best_model_name]\n",
                    "\n",
                    "final_pipeline = Pipeline([\n",
                    "    ('raw_preprocess', FunctionTransformer(raw_preprocess_transformer)),\n",
                    "    ('preprocessor', preprocessor),\n",
                    "    ('regressor', best_model_instance)\n",
                    "])\n",
                    "\n",
                    "# Train on whole dataset\n",
                    "final_pipeline.fit(X, y)\n",
                    "\n",
                    "# Ensure directory exists\n",
                    "os.makedirs('../models', exist_ok=True)\n",
                    "joblib.dump(final_pipeline, '../models/laptop_price_pipeline.pkl')\n",
                    "print(\"Trained pipeline successfully saved to ../models/laptop_price_pipeline.pkl\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs('../notebooks', exist_ok=True)
    with open('../notebooks/Model_Training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Model_Training.ipynb generated.")

if __name__ == '__main__':
    # Go to script folder parent to run properly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    create_eda_notebook()
    create_model_training_notebook()
