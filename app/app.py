"""
Main Streamlit application for Laptop Price Predictor.
Contains Home, Predict Price, Dashboard, and About pages.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict import predict_price
from utils.feature_importance import get_feature_importances, plot_importance_plotly

# Set Page Config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C5B7B, #C06C84, #F67280);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F67280;
    }
    
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: #6C5B7B;
        margin: 10px 0;
    }
    
    .price-range {
        font-size: 1.1rem;
        color: #F8B195;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset for dashboard
@st.cache_data
def load_data():
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'laptop_data.csv'
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, encoding='latin-1')
    return pd.DataFrame()

df = load_data()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #F67280;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to:", ["Home", "Predict Price", "Dashboard"])

# ----------------- HOME PAGE -----------------
if page == "Home":
    st.markdown("<h1 class='main-title'>Laptop Price Valuation Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Instant, intelligent laptop market-value appraisal</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Find the True Value of Any Laptop Configuration
        Welcome to the premium **Laptop Valuation Portal**. Whether you are buying a new device, selling your current laptop, or tracking market configurations, this application provides an immediate, evidence-based valuation.
        
        #### How it helps you:
        *   **Smart Buying**: Avoid overpaying. Run a quick prediction on any laptop specs to verify if a retail price matches the fair market value.
        *   **Smart Selling**: Appraise your device accurately. Price your laptop competitively on online marketplaces to sell faster.
        *   **Market Intelligence**: Explore pricing distributions and component dynamics interactively in our dashboard.
        
        Use the sidebar navigation to head to **Predict Price** to value a laptop, or open the **Dashboard** to view interactive market insights.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Get Started", use_container_width=True):
            st.info("Please select 'Predict Price' in the sidebar navigation to estimate a price!")
            
    with col2:
        st.image("https://images.unsplash.com/photo-1593642632823-8f785ba67e45?q=80&w=600&auto=format&fit=crop", 
                 caption="Get real-time market value appraisals based on hardware specifications.", use_container_width=True)

# ------------- PREDICT PRICE PAGE -------------
elif page == "Predict Price":
    st.markdown("<h1 class='main-title'>Predict Laptop Price</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Select specifications to estimate the market value</p>", unsafe_allow_html=True)
    
    if df.empty:
        st.error("Dataset not found. Please train the model and save the data in the appropriate folder first.")
    else:
        # Form for specifications
        with st.form("laptop_spec_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### General Info")
                company = st.selectbox("Company / Brand", sorted(df['Company'].unique()))
                type_name = st.selectbox("Laptop Type", sorted(df['TypeName'].unique()))
                weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1)
                op_sys = st.selectbox("Operating System", sorted(df['OpSys'].unique()))
                
            with col2:
                st.markdown("### Screen & CPU")
                inches = st.slider("Screen Size (Inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
                
                # Screen resolution options
                res_option = st.selectbox("Screen Resolution", [
                    "Full HD 1920x1080",
                    "IPS Panel Full HD 1920x1080",
                    "IPS Panel Full HD / Touchscreen 1920x1080",
                    "IPS Panel Retina Display 2560x1600",
                    "IPS Panel Retina Display 2880x1800",
                    "IPS Panel 4K Ultra HD / Touchscreen 3840x2160",
                    "IPS Panel 4K Ultra HD 3840x2160",
                    "Touchscreen / Quad HD+ 3200x1800",
                    "Quad HD+ 3200x1800",
                    "1366x768",
                    "Touchscreen 1366x768",
                    "1440x900",
                    "1600x900",
                    "2560x1440"
                ])
                
                # CPU Options
                cpu_brand = st.selectbox("CPU Brand", [
                    "Intel Core i7", "Intel Core i5", "Intel Core i3",
                    "Intel Celeron", "Intel Pentium", "AMD Ryzen",
                    "AMD Processor", "Other Intel Processor", "Other Processor"
                ])
                cpu_speed = st.slider("CPU Clock Speed (GHz)", min_value=0.9, max_value=4.0, value=2.5, step=0.1)
                
            with col3:
                st.markdown("### Memory & GPU")
                ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
                ssd = st.selectbox("SSD Capacity (GB)", [0, 8, 16, 32, 64, 128, 180, 246, 256, 512, 1000, 2000], index=8)
                hdd = st.selectbox("HDD Capacity (GB)", [0, 128, 256, 512, 1000, 2000], index=0)
                flash = st.selectbox("Flash Storage (GB)", [0, 16, 32, 64, 128, 256, 512], index=0)
                gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD", "ARM"])
                
            submit_button = st.form_submit_button("Predict Price", use_container_width=True)
            
        if submit_button:
            # Reconstruct the string columns to match original dataset format
            # Screen resolution details
            touchscreen = "Touchscreen" if "Touchscreen" in res_option else ""
            ips = "IPS Panel" if "IPS" in res_option else ""
            width, height = 1920, 1080 # default fallback
            for match in res_option.split():
                if 'x' in match:
                    parts = match.split('x')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        width, height = int(parts[0]), int(parts[1])
                        
            # CPU string
            cpu_str = f"{cpu_brand} {cpu_speed}GHz"
            
            # Ram string
            ram_str = f"{ram}GB"
            
            # Weight string
            weight_str = f"{weight}kg"
            
            # Memory string
            memory_parts = []
            if ssd > 0:
                memory_parts.append(f"{ssd}GB SSD")
            if hdd > 0:
                hdd_str = f"{hdd//1000}TB HDD" if hdd >= 1000 else f"{hdd}GB HDD"
                memory_parts.append(hdd_str)
            if flash > 0:
                memory_parts.append(f"{flash}GB Flash Storage")
            
            memory_str = " + ".join(memory_parts) if memory_parts else "128GB SSD"
            
            # GPU string
            gpu_str = f"{gpu_brand} Graphics"
            
            # Reconstruct dict
            input_dict = {
                "Company": company,
                "Product": "Generic Laptop", # Ignored by preprocess drop
                "TypeName": type_name,
                "Inches": inches,
                "ScreenResolution": res_option,
                "Cpu": cpu_str,
                "Ram": ram_str,
                "Memory": memory_str,
                "Gpu": gpu_str,
                "OpSys": op_sys,
                "Weight": weight_str
            }
            
            with st.spinner("Calculating estimated price..."):
                try:
                    pred_price, low_price, high_price = predict_price(input_dict)
                    
                    # Display Results
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    eur_to_inr = 92.0
                    pred_price_inr = pred_price * eur_to_inr
                    low_price_inr = low_price * eur_to_inr
                    high_price_inr = high_price * eur_to_inr
                    
                    st.markdown(f"""
                    <div class='card'>
                        <div class='prediction-title'>Estimated Market Price</div>
                        <div class='price-display'>€ {pred_price:,.2f} / ₹ {pred_price_inr:,.2f}</div>
                        <div class='price-range'>Estimated price range: € {low_price:,.2f} - € {high_price:,.2f} (₹ {low_price_inr:,.2f} - ₹ {high_price_inr:,.2f})</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Feature Importance chart alongside prediction
                    st.markdown("### Top Factors Influencing Price Predictions")
                    try:
                        df_imp = get_feature_importances()
                        fig_imp = plot_importance_plotly(df_imp)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as fe:
                        st.warning(f"Could not load feature importance chart: {str(fe)}")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# ----------------- DASHBOARD PAGE -----------------
elif page == "Dashboard":
    st.markdown("<h1 class='main-title'>Laptop Price Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Interactive exploratory insights on laptop configurations and pricing</p>", unsafe_allow_html=True)
    
    if df.empty:
        st.error("No dataset found. Please place `laptop_data.csv` inside the `data/` folder.")
    else:
        # Preprocess Ram and Weight for visualization
        df_viz = df.copy()
        df_viz['Ram_GB'] = df_viz['Ram'].str.replace('GB', '', regex=False).astype(int)
        df_viz['Weight_kg'] = df_viz['Weight'].str.replace('kg', '', regex=False).astype(float)
        
        # Row 1: KPI Metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Laptops in Dataset", len(df))
        
        eur_to_inr = 92.0
        avg_price_eur = df['Price_euros'].mean()
        max_price_eur = df['Price_euros'].max()
        avg_price_inr = avg_price_eur * eur_to_inr
        max_price_inr = max_price_eur * eur_to_inr
        
        kpi2.metric("Average Laptop Price", f"€ {avg_price_eur:,.2f} / ₹ {avg_price_inr:,.2f}")
        kpi3.metric("Max Laptop Price", f"€ {max_price_eur:,.2f} / ₹ {max_price_inr:,.2f}")
        kpi4.metric("Unique Brands", len(df['Company'].unique()))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Price and RAM Distributions
        col1, col2 = st.columns(2)
        
        with col1:
            fig_price = px.histogram(df_viz, x='Price_euros', marginal='box', 
                                     title='Overall Price Distribution',
                                     color_discrete_sequence=['#C06C84'])
            st.plotly_chart(fig_price, use_container_width=True)
            
        with col2:
            ram_counts = df_viz['Ram_GB'].value_counts().sort_index().reset_index()
            ram_counts.columns = ['Ram_GB', 'Count']
            fig_ram = px.bar(ram_counts, x='Ram_GB', y='Count', 
                             title='RAM Capacity Count (GB)',
                             color='Count', color_continuous_scale='Sunset')
            fig_ram.update_layout(xaxis=dict(type='category'))
            st.plotly_chart(fig_ram, use_container_width=True)
            
        # Row 3: Brand & Laptop Type
        col3, col4 = st.columns(2)
        
        with col3:
            brand_counts = df_viz['Company'].value_counts().reset_index()
            brand_counts.columns = ['Company', 'Count']
            fig_brand = px.bar(brand_counts, x='Company', y='Count', 
                               title='Total Laptop Count by Brand',
                               color='Count', color_continuous_scale='Purples')
            st.plotly_chart(fig_brand, use_container_width=True)
            
        with col4:
            type_counts = df_viz['TypeName'].value_counts().reset_index()
            type_counts.columns = ['TypeName', 'Count']
            fig_type = px.pie(type_counts, names='TypeName', values='Count', 
                              title='Laptop Categories Type Share',
                              color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_type, use_container_width=True)
            
        # Row 4: Average prices by Brand and CPU/GPU
        st.markdown("### Average Prices by Component Brands")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            avg_price_brand = df_viz.groupby('Company')['Price_euros'].mean().reset_index().sort_values(by='Price_euros', ascending=False)
            fig_avg_brand = px.bar(avg_price_brand, x='Company', y='Price_euros', 
                                   title='Average Price by Company',
                                   color='Price_euros', color_continuous_scale='Reds')
            st.plotly_chart(fig_avg_brand, use_container_width=True)
            
        with col6:
            # Extract CPU brand for average price viz
            def quick_cpu(x):
                if 'i7' in x: return 'Intel Core i7'
                if 'i5' in x: return 'Intel Core i5'
                if 'i3' in x: return 'Intel Core i3'
                if 'Celeron' in x: return 'Intel Celeron'
                if 'Pentium' in x: return 'Intel Pentium'
                if 'AMD' in x: return 'AMD Processor'
                return 'Other Processor'
                
            df_viz['CPU Brand'] = df_viz['Cpu'].apply(quick_cpu)
            avg_price_cpu = df_viz.groupby('CPU Brand')['Price_euros'].mean().reset_index().sort_values(by='Price_euros', ascending=False)
            fig_avg_cpu = px.bar(avg_price_cpu, x='CPU Brand', y='Price_euros', 
                                 title='Average Price by CPU Brand',
                                 color='Price_euros', color_continuous_scale='Oranges')
            st.plotly_chart(fig_avg_cpu, use_container_width=True)
            
        with col7:
            df_viz['GPU Brand'] = df_viz['Gpu'].apply(lambda x: x.split()[0])
            avg_price_gpu = df_viz.groupby('GPU Brand')['Price_euros'].mean().reset_index().sort_values(by='Price_euros', ascending=False)
            fig_avg_gpu = px.bar(avg_price_gpu, x='GPU Brand', y='Price_euros', 
                                 title='Average Price by GPU Brand',
                                 color='Price_euros', color_continuous_scale='Teal')
            st.plotly_chart(fig_avg_gpu, use_container_width=True)
