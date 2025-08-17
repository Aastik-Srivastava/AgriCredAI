import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Capital One AgriCred AI - Agricultural Credit Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Aastik-Srivastava/AgriCredAI',
        'Report a bug': 'https://github.com/Aastik-Srivastava/AgriCredAI/issues',
        'About': "Capital One AgriCred AI - Revolutionizing agricultural lending with AI"
    }
)

import pandas as pd
import numpy as np
import joblib  # For loading machine learning models
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import requests
import json
import re

import random
import io, wave, os
from sklearn.ensemble import RandomForestRegressor



# Custom modules (assuming these are in your project directory)
from agentic_ai_demo import agentic_ai_demo
from advanced_data_pipeline import AdvancedDataPipeline
from advanced_ml_model import AdvancedCreditModel
from weather_alert_system import WeatherAlertSystem, setup_alerts_table
from config import (
    MODEL_PATH, SCALER_PATH,  # Paths for ML model and scaler
    WEATHER_API_KEY, MARKET_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, WEATHER_UNITS, ALERT_CHECK_INTERVAL # Weather API and database config
)
# Optional speech libs
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = KaldiRecognizer = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from config import VOSK_MODEL_PATH
except Exception:
    VOSK_MODEL_PATH = None

# Safe defaults if not defined elsewhere
try:
    CREDIT_PRICE_USD
except NameError:
    CREDIT_PRICE_USD = 12.0  # $/tCO2e (demo)
try:
    USD_TO_INR
except NameError:
    USD_TO_INR = 83.0
try:
    CAR_EQUIV_TON
except NameError:
    CAR_EQUIV_TON = 4.6   # ~tCO2 avoided per car/year (very rough demo figure)
try:
    TREE_EQUIV_TON
except NameError:
    TREE_EQUIV_TON = 0.021  # ~21 kg CO2 per tree/year (demo)

from credit_db_maker import store_credit_transaction, DB_PATH, CREDIT_PRICE_USD, USD_TO_INR, CAR_EQUIV_TON, TREE_EQUIV_TON

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = AdvancedDataPipeline()


# Global styling
st.markdown("""
<style>
    /* Global base styles (Light mode by default) */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5a8a 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        color: #333333; 
    }
    
    .risk-low { border-left: 5px solid #28a745; }
    .risk-medium { border-left: 5px solid #ffc107; }
    .risk-high { border-left: 5px solid #dc3545; }
    
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #333333; /* Default text color for light mode */
    }
            
    .financier-insight {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
        color: #333333;
    }
    
    .portfolio-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    /* Welcome screen specific light mode styles (using classes for better targeting) */
    .welcome-container {
        text-align: center;
        padding: 2rem;
        /* Default text color assumed from Streamlit's base light theme if not specified */
    }
    
    .welcome-info-box {
        background: #f0f2f6; 
        padding: 1.5rem;
        border-radius: 10px;
        color: #333333; /* Dark text for light mode box */
    }

   

    /* Dark Mode Styling */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #1A3C5B 0%, #264A6A 100%); 
            color: #f0f2f6; 
        }
        
        .metric-card {
            background: #262626; 
            color: #f0f2f6; 
            box-shadow: 0 2px 4px rgba(255,255,255,0.1); 
        }
        
        .risk-low { border-left: 5px solid #6edc86; } 
        .risk-medium { border-left: 5px solid #ffd75e; } 
        .risk-high { border-left: 5px solid #ff7b7b; } 
        
        .sidebar-logo {
            background: #262626; 
            color: #f0f2f6; /* Light text for dark mode sidebar logo */
        }
                
        .financier-insight {
            background: #1f2e46; 
            border-left: 4px solid #5a87be; 
            color: #f0f2f6; 
        }
        
        .portfolio-summary {
            background: linear-gradient(135deg, #4b5f88 0%, #5d4679 100%); 
            color: #f0f2f6; 
        }

        /* Welcome Screen specific dark mode styles */
        .welcome-container {
             color: #f0f2f6; /* Light text color */
        }
        .welcome-container h3,
        .welcome-container h4 {
            color: #90ee90; /* Lighter green for headings in dark theme */
        }
        .welcome-container p,
        .welcome-container ul li {
            color: #f0f2f6; /* Light text color for paragraphs and list items */
        }
        
        .welcome-info-box {
             background: #1f2e46; 
             color: #f0f2f6; 
        }
        .welcome-info-box h4 {
            color: #90ee90; /* Lighter green for inner box heading in dark theme */
        }
    }
</style>
""", unsafe_allow_html=True)

def display_main_header():
    """Display the main platform header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Capital One AgriCred AI Platform</h1>
        <h3>Advanced Agricultural Credit Intelligence & Risk Management</h3>
        <p>Empowering financial institutions with AI-driven insights for agricultural lending</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Enhanced sidebar with financier focus"""
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <h2>🏦 Capital One</h2>
        <p><strong>AgriCred AI</strong></p>
        <p style="font-size: 12px; color: #666;">Agricultural Lending Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📊 Navigation Dashboard")
    
    page_options = [
        "🏠 Executive Summary",
        "📊 Portfolio Analytics", 
        "🎯 Credit Risk Scoring",
        "🤖 Agentic AI Intelligence",
        "🌦️ Weather Risk Monitor",
        "💹 Market Intelligence",
        "📈 Performance Analytics",
        "⚙️ System Configuration"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Select Dashboard",
        page_options,
        help="Choose your dashboard view"
    )
    
    # Real-time metrics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Live Metrics")
    
   # In display_sidebar(), remove random metrics and add placeholders:
    metrics = st.session_state.pipeline.calculate_and_store_portfolio_metrics()
    st.sidebar.metric("Portfolio Value", f"₹{metrics['total_portfolio']/1e7:.1f}Cr")
    st.sidebar.metric("Active Loans", f"{metrics['total_loans']:,}")
    st.sidebar.metric("Default Rate", f"{metrics['default_rate']:.1f}%")
    st.sidebar.metric("Avg Credit Score", f"{int(metrics['avg_credit_score'])}")

    # Weather alerts
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌦️ Weather Alerts")
    alert_count = random.randint(5, 15)
    st.sidebar.error(f"⚠️ {alert_count} High Risk Alerts")
    st.sidebar.warning("🌧️ Heavy Rain Warning: Maharashtra")
    st.sidebar.info("🌡️ Temperature Alert: Punjab")
    
    return selected_page


@st.cache_data
def fetch_market_prices():
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        df = pd.DataFrame(records)
        return df
    else:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except:
        return None, None




@st.cache_resource
def initialize_data_pipeline():
    """Initialize data pipeline"""
    return AdvancedDataPipeline()

@st.cache_resource
def get_alert_system():
    setup_alerts_table()
    return WeatherAlertSystem()


API_URL = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={MARKET_API_KEY}&format=json&limit=100"

# List of cities and coordinates
CITIES = [
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Goa", "lat": 15.2993, "lon": 74.1240},
]
CITY_COORDS = {
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
    "Goa": (15.2993, 74.1240),
    "Udupi": (13.3522, 74.7919),
}
def performance_analytics():
    """Performance analytics and reporting"""
    st.markdown("## 📈 Performance Analytics & Reporting")
    
    # Generate performance data
    months = pd.date_range(start='2024-09-01', end='2025-08-31', freq='MS')
    performance_data = {
        'Month': months,
        'Revenue (₹Cr)': np.cumsum(np.random.normal(8, 1, 12)) + 85,
        'Profit (₹Cr)': np.cumsum(np.random.normal(2, 0.5, 12)) + 25,
        'Cost of Funds (%)': np.random.normal(0, 0.1, 12) + 7.2,
        'NPA Ratio (%)': np.maximum(0, np.cumsum(np.random.normal(0, 0.2, 12)) + 4.1),
        'ROA (%)': np.random.normal(0, 0.2, 12) + 2.8,
        'New Loans': np.random.poisson(380, 12)
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    # Key performance indicators
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        latest_revenue = df_perf['Revenue (₹Cr)'].iloc[-1]
        prev_revenue = df_perf['Revenue (₹Cr)'].iloc[-2] 
        revenue_change = (latest_revenue - prev_revenue) / prev_revenue * 100
        st.metric("Monthly Revenue", f"₹{latest_revenue:.1f}Cr", f"{revenue_change:+.1f}%")
    
    with col2:
        latest_profit = df_perf['Profit (₹Cr)'].iloc[-1]
        prev_profit = df_perf['Profit (₹Cr)'].iloc[-2]
        profit_change = (latest_profit - prev_profit) / prev_profit * 100
        st.metric("Monthly Profit", f"₹{latest_profit:.1f}Cr", f"{profit_change:+.1f}%")
    
    with col3:
        latest_npa = df_perf['NPA Ratio (%)'].iloc[-1]
        st.metric("NPA Ratio", f"{latest_npa:.2f}%", help="Non-performing assets ratio")
    
    with col4:
        latest_roa = df_perf['ROA (%)'].iloc[-1]
        st.metric("ROA", f"{latest_roa:.2f}%", help="Return on assets")
    
    with col5:
        latest_loans = df_perf['New Loans'].iloc[-1]
        st.metric("New Loans", f"{latest_loans}", help="New loans this month")
    
    # Performance trends
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue and profit trend
        fig_revenue = px.line(
            df_perf,
            x='Month',
            y=['Revenue (₹Cr)', 'Profit (₹Cr)'],
            title='Revenue & Profit Trends'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # NPA and ROA trend
        fig_ratios = px.line(
            df_perf,
            x='Month', 
            y=['NPA Ratio (%)', 'ROA (%)'],
            title='Key Financial Ratios'
        )
        st.plotly_chart(fig_ratios, use_container_width=True)
    
    # Loan disbursement trend
    fig_loans = px.bar(
        df_perf,
        x='Month',
        y='New Loans',
        title='Monthly New Loan Disbursements'
    )
    st.plotly_chart(fig_loans, use_container_width=True)

# --- Load Data from DB ---
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM credits", conn)
    conn.close()
    return df

def get_weather(lat, lon):
    """Fetch live weather data for given lat/lon."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params)
    return r.json() if r.status_code == 200 else None

def parse_weather_data(weather_json):
    """Convert raw weather data into readable text and alerts."""
    if not weather_json:
        return "", []

    city = weather_json["name"]
    temp = weather_json["main"]["temp"]
    humidity = weather_json["main"]["humidity"]
    wind = weather_json["wind"]["speed"]
    desc = weather_json["weather"][0]["description"].title()

    report = f"**{city}**: {desc}, 🌡 {temp}°C, 💧 Humidity {humidity}%, 💨 Wind {wind} m/s"

    # Basic alert rules
    alerts = []
    if temp > 35:
        alerts.append(("Heatwave Risk", "High"))
    if humidity > 90 and "rain" in desc.lower():
        alerts.append(("Heavy Rain Alert", "Medium"))
    if wind > 15:
        alerts.append(("High Wind Alert", "High"))
    if not alerts:
        alerts.append(("All Clear", "Low"))

    return report, alerts

def display_weather_reports():

    text_color = st.get_option("theme.textColor")
    background_color = st.get_option("theme.backgroundColor")

    st.markdown("### 📄 Latest Weather Reports")
    cols = st.columns(2)  # Two-column layout for compactness
    
    for i, city in enumerate(CITIES):
        data = get_weather(city["lat"], city["lon"])
        if data:
            report, _ = parse_weather_data(data)
            
            # Extract components for styling
            name = data["name"]
            desc = data["weather"][0]["description"].title()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]

            with cols[i % 2]:
                st.markdown(f"""
                <div style='
                    background-color: {background_color}; 
                    color: {text_color};
                    border-radius: 10px; 
                    padding: 12px; 
                    margin-bottom: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>
                    <h4 style='margin-bottom: 4px; color: {text_color};'>{name}</h4>
                    <p style='margin: 0; font-size: 15px; color: {text_color};'>{desc}</p>
                    <p style='margin: 0; color: {text_color};'>🌡 {temp}°C&nbsp;&nbsp;💧 {humidity}%&nbsp;&nbsp;💨 {wind} m/s</p>
                </div>
                """, unsafe_allow_html=True)

def display_alerts(alerts_feed):
    text_color = st.get_option("theme.textColor")
    background_color = st.get_option("theme.backgroundColor")
    st.subheader("📡 Live Weather Alerts Feed")
    
    severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    for alert in alerts_feed:
        st.markdown(f"""
        <div style='
            background-color: {background_color}; 
            border-radius: 8px; 
            padding: 8px 12px; 
            margin-bottom: 6px;
            border-left: 5px solid {"#dc3545" if alert["severity"]=="High" else "#ffc107" if alert["severity"]=="Medium" else "#28a745"};
        '>
            <strong>{severity_color[alert['severity']]} {alert['alert']}</strong> — {alert['city']}
        </div>
        """, unsafe_allow_html=True)




def fetch_weather(lat, lon):
    """Return a simplified weather summary and risk proxy dict."""
    try:
        url = f"https://api.openweathermap.org/data/3.0/onecall?"
        params = {
            "lat": float(lat),
            "lon": float(lon),
            "exclude": "minutely,hourly,alerts",
            "units": "metric",
            "appid": WEATHER_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # current + daily summary
        current = data.get("current", {})
        daily = data.get("daily", [])

        # compute 7-day rainfall sum (if available)
        rainfall_7day = 0.0
        for d in daily[:7]:
            # openweather may have d.get('rain') or daily precipitation array
            rainfall_7day += d.get("rain", 0.0) if d.get("rain") else 0.0

        # simple risk proxies (tunable thresholds)
        temp = current.get("temp")
        humidity = current.get("humidity")
        # frost_risk: if min temp in next 3 days < 3C
        frost_risk = 0.0
        for d in daily[:3]:
            if d.get("temp", {}).get("min") is not None and d["temp"]["min"] < 3.0:
                frost_risk = max(frost_risk, 0.9)  # high frost risk
        # drought proxy: low cumulative rainfall compared to threshold
        drought_risk = 0.0
        if rainfall_7day < 10:  # <10 mm in 7 days -> drought-ish (tweak by crop/season)
            drought_risk = 0.8
        elif rainfall_7day < 30:
            drought_risk = 0.4

        return {
            "temperature": temp,
            "humidity": humidity,
            "rainfall_7day": rainfall_7day,
            "frost_risk": frost_risk,
            "drought_risk": drought_risk,
            "raw": data
        }

    except Exception as e:
        st.warning(f"Weather fetch failed: {e}")
        # fallback safe defaults (so rest of pipeline works)
        return {
            "temperature": None,
            "humidity": None,
            "rainfall_7day": None,
            "frost_risk": 0.0,
            "drought_risk": 0.0,
            "raw": {}
        }
    

def credit_risk_scoring_dashboard():
    # Header
    st.markdown("# 🏦 Agricultural Credit Risk Assessment")
    st.markdown("### AI-Powered Credit Scoring for Agricultural Lending")
    st.markdown("---")
    
    pipeline = initialize_data_pipeline()

    # Load model artifacts
    try:
        model = joblib.load('advanced_credit_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model_type = "xgboost"  # Set based on your best model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Please ensure model files are present: advanced_credit_model.pkl, feature_scaler.pkl, feature_columns.pkl")
        return
    
    # Complete feature defaults
    defaults = {
    'farmer_age': 40,                    # Younger, more tech-savvy
    'education_level': 4,                # Above average education
    'family_size': 4,                    # Smaller household obligations
    'land_size': 3.0,                    # Moderately large farm
    'crop_type_encoded': 2,              # Wheat (stable commodity)
    'irrigation_access': 1,              # ✅ Has irrigation
    'current_temperature': 28.0,         # Optimal growing temp
    'current_humidity': 60,              # Ideal humidity
    'temperature_stress': 0.1,           # Low stress
    'humidity_stress': 0.1,              # Low stress
    'drought_risk_7days': 0.1,           # Low drought risk
    'frost_risk_7days': 0.01,            # Almost zero frost risk
    'excess_rain_risk': 0.05,            # Very low flood risk
    'price_volatility': 0.1,             # Stable prices
    'nearest_mandi_distance': 10.0,      # Close to market
    'connectivity_index': 0.8,           # Strong connectivity
    'input_cost_index': 0.3,             # Lower input costs
    'loan_to_land_ratio': 0.2,           # Conservative borrowing
    'debt_to_income_ratio': 0.2,         # Low debt burden
    'payment_history_score': 0.95,       # Excellent history
    'yield_consistency': 0.9,            # Very consistent yields
    'soil_health_index': 0.9,            # Very healthy soil
    'nutrient_deficiency_risk': 0.05,    # Negligible nutrient risk
    'insurance_coverage': 1,             # ✅ Insured
    'cooperative_membership': 1,         # ✅ Member
    'technology_adoption': 0.8,          # High tech use
    'diversification_index': 0.7,        # Well diversified
    'electricity_reliability': 0.9,      # Very reliable power
    'mobile_network_strength': 0.9,      # Excellent connectivity
    'bank_branch_distance': 5.0,         # Very close to bank
    # And for the rest, use similarly low‐risk values:
    'seasonal_rainfall_deviation': 0.0,
    'historical_drought_frequency': 0,
    'climate_change_vulnerability': 0.1,
    'current_price': 200000.0,
    'market_demand_index': 0.8,
    'export_potential': 0.7,
    'storage_price_premium': 0.1,
    'price_trend': 0.05,
    'savings_to_income_ratio': 0.2,
    'credit_utilization': 0.2,
    'number_of_credit_sources': 2,
    'informal_lending_dependency': 0.1,
    'road_quality_index': 0.9,
    'mechanization_level': 0.8,
    'seed_quality_index': 0.9,
    'fertilizer_usage_efficiency': 0.9,
    'pest_management_score': 0.8,
    'organic_farming_adoption': 0.3,
    'precision_agriculture_usage': 0.7,
    'eligible_schemes_count': 3,
    'subsidy_utilization': 0.8,
    'msp_eligibility': 1,
    'kisan_credit_card': 1,
    'government_training_participation': 0.8,
    'community_leadership_role': 1,
    'social_capital_index': 0.8,
    'extension_service_access': 0.8,
    'peer_learning_participation': 0.8,
    'labor_availability': 0.8,
    'storage_access': 1,
    'transport_cost_burden': 0.2,
    'supply_chain_integration': 0.8,
    'disaster_preparedness': 0.8,
    'alternative_income_sources': 0.7,
    'livestock_ownership': 1
}

    # Input form in sidebar
    st.markdown("## 📝 Farmer Assessment Form")
    st.markdown("*Enter key information for credit evaluation*")
    
    # Farmer details
    farmer_name = st.text_input("👤 Farmer Name", "Rajesh Kumar")
    monthly_income = st.number_input("💰 Monthly Income (₹)", min_value=5000, max_value=200000, value=25000)
    
    # Key risk factors
    user_inputs = {}
    
    with st.expander("🏦 Financial Information", expanded=True):
        user_inputs['payment_history_score'] = st.slider("Payment History Score", 0.1, 1.0, 0.85, help="Track record of loan repayments")
        user_inputs['debt_to_income_ratio'] = st.slider("Debt to Income Ratio", 0.0, 2.0, 0.4, help="Monthly debt payments / Monthly income")
        user_inputs['savings_to_income_ratio'] = st.slider("Savings Rate", 0.0, 0.5, 0.1, help="Percentage of income saved monthly")
    
    with st.expander("🌾 Agricultural Details", expanded=True):
        user_inputs['land_size'] = st.number_input("Land Size (hectares)", 0.5, 20.0, 2.0, help="Total cultivated land")
        user_inputs['yield_consistency'] = st.slider("Yield Consistency", 0.3, 1.0, 0.7, help="Reliability of crop yields")
        user_inputs['irrigation_access'] = st.radio("Irrigation Access?", [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['soil_health_index'] = st.slider("Soil Health", 0.2, 1.0, 0.75, help="Soil quality and fertility")
    
    with st.expander("🌦️ Climate & Weather Risks", expanded=False):
        user_inputs['drought_risk_7days'] = st.slider("7-day Drought Risk", 0.0, 1.0, 0.3)
        user_inputs['price_volatility'] = st.slider("Price Volatility", 0.05, 0.8, 0.2, help="Market price fluctuation")
    
    with st.expander("🤝 Support Systems", expanded=False):
        user_inputs['cooperative_membership'] = st.radio("Cooperative Member?", [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['insurance_coverage'] = st.radio("Crop Insurance?", [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['technology_adoption'] = st.slider("Technology Adoption", 0.1, 0.95, 0.5, help="Use of modern farming techniques")
        user_inputs['diversification_index'] = st.slider("Crop Diversification", 0.1, 0.9, 0.4, help="Variety of crops grown")
    
    # Assessment button
    assess_button = st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True)
    
    # Main content area
    if not assess_button:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>🌾 Agricultural Credit Assessment</h3>
                <p>Complete the farmer assessment form and click 
                <strong>"Assess Credit Risk"</strong> to generate a comprehensive credit evaluation.</p>
                <br>
            </div>
            """, unsafe_allow_html=True)


    else:
        # Build prediction input
        features = defaults.copy()
        features.update(user_inputs)
        input_list = [features[feat] for feat in feature_columns]
        input_df = pd.DataFrame([input_list], columns=feature_columns)
        
        try:
            # Make prediction
            input_scaled = scaler.transform(input_df)
            pred_prob = model.predict_proba(input_scaled)[0][1]
            credit_score = int((1 - pred_prob) * 750 + 250)
            
            # Professional Results Display
            st.markdown("---")
            st.subheader(f"📊 Comprehensive Assessment for {farmer_name}")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Credit Score", credit_score, help="FICO-style score (250-1000)")
            with col2:
                st.metric("⚠️ Default Risk", f"{pred_prob:.1%}", help="Probability of default")
            with col3:
                if pred_prob < 0.4:
                    st.success("✅ APPROVE")
                    recommendation = "APPROVE"
                elif pred_prob < 0.7:
                    st.warning("⚠️ REVIEW")
                    recommendation = "REVIEW"
                else:
                    st.error("❌ REJECT")
                    recommendation = "REJECT"
            with col4:
                loan_capacity = int(monthly_income * 12 * 3 * (1 - pred_prob))
                st.metric("💰 Max Loan Capacity", f"₹{loan_capacity:,}", help="Recommended maximum loan amount")
            
            # Risk breakdown
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🧠 AI Decision Explanation")
                
                # Generate risk factors based on inputs
                positive_factors = []
                negative_factors = []
                recommendations = []
                
                # Analyze positive factors
                if user_inputs['payment_history_score'] > 0.8:
                    positive_factors.append("Excellent payment history")
                if user_inputs['debt_to_income_ratio'] < 0.3:
                    positive_factors.append("Low debt burden")
                if user_inputs['yield_consistency'] > 0.7:
                    positive_factors.append("Consistent crop yields")
                if user_inputs['irrigation_access']:
                    positive_factors.append("Access to irrigation")
                if user_inputs['insurance_coverage']:
                    positive_factors.append("Crop insurance coverage")
                if user_inputs['cooperative_membership']:
                    positive_factors.append("Member of farming cooperative")
                
                # Analyze risk factors
                if user_inputs['debt_to_income_ratio'] > 0.5:
                    negative_factors.append("High debt-to-income ratio")
                if user_inputs['price_volatility'] > 0.3:
                    negative_factors.append("High market price volatility")
                if user_inputs['drought_risk_7days'] > 0.5:
                    negative_factors.append("Significant drought risk")
                if user_inputs['diversification_index'] < 0.3:
                    negative_factors.append("Limited crop diversification")
                if user_inputs['technology_adoption'] < 0.3:
                    negative_factors.append("Low technology adoption")
                
                # Generate recommendations
                if not user_inputs['insurance_coverage']:
                    recommendations.append("Consider purchasing crop insurance")
                if user_inputs['diversification_index'] < 0.5:
                    recommendations.append("Increase crop diversification")
                if user_inputs['technology_adoption'] < 0.5:
                    recommendations.append("Adopt modern farming technologies")
                if user_inputs['debt_to_income_ratio'] > 0.4:
                    recommendations.append("Focus on debt reduction strategies")
                
                # Display factors
                if positive_factors:
                    st.markdown("**✅ Positive Factors:**")
                    for factor in positive_factors[:5]:
                        st.markdown(f"• {factor}")
                
                if negative_factors:
                    st.markdown("**⚠️ Risk Factors:**")
                    for factor in negative_factors[:5]:
                        st.markdown(f"• {factor}")
                
                if recommendations:
                    st.markdown("**📋 Recommendations:**")
                    for rec in recommendations[:3]:
                        st.markdown(f"• {rec}")
                
                if not positive_factors and not negative_factors:
                    st.info("Assessment based on standard agricultural lending criteria.")
            
            with col2:
                # Risk gauge
                risk_value = pred_prob * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Level", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if risk_value > 60 else "orange" if risk_value > 30 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance visualization
            st.markdown("---")
            st.subheader("📈 Model Feature Analysis")
            
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Create SHAP summary
                feature_impact = []
                for i, (feature, shap_val, feat_val) in enumerate(zip(feature_columns, shap_values[1][0], input_df.iloc[0].values)):
                    feature_impact.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'Impact': shap_val,
                        'Value': feat_val
                    })
                
                # Sort by absolute impact
                feature_impact.sort(key=lambda x: abs(x['Impact']), reverse=True)
                
                # Display top 10 features
                impact_df = pd.DataFrame(feature_impact[:10])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Influential Features:**")
                    st.dataframe(impact_df, hide_index=True)
                
                with col2:
                    # Bar chart of feature impacts
                    fig_bar = px.bar(
                        impact_df, 
                        x='Impact', 
                        y='Feature',
                        orientation='h',
                        title="Feature Impact on Risk Score",
                        color='Impact',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
            except Exception:
                # Fallback to model feature importance if SHAP fails
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': [f.replace('_', ' ').title() for f in feature_columns],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Model Feature Importance"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature analysis not available for this model type.")

            user_crop = st.text_input("Main Crop (e.g. wheat, cotton, rice)", "all")
            user_state = st.text_input("State (e.g. Maharashtra, Bihar)", "all")
            user_land_size = user_inputs.get('land_size', 2.0)  # Already in your user inputs block
            policy_advisor_with_keyword_search(user_land_size, user_crop, user_state)


        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please check if all required model files are present and properly trained.")

        


def map_land_size_category(hectares):
    if hectares < 0.4:
        return 'marginal'
    elif hectares < 0.8:
        return 'small'
    elif hectares < 4:
        return 'medium'
    else:
        return 'large'

def policy_advisor_with_keyword_search(land_size_hectares, crop_keyword='all', state_keyword='all'):
    st.header('🏛️ Dynamic Government Policy Advisor')
    st.markdown("""
        **Relevant government schemes and policies based on your farm profile and location.**
    """)

    try:
        with open('myschemes_full.json', 'r', encoding='utf-8') as f:
            schemes = json.load(f)
    except FileNotFoundError:
        st.error('❌ `myschemes_full.json` not found. Please ensure the file is in the app directory.')
        return
    
    land_category = map_land_size_category(land_size_hectares).lower()
    crop_keyword = crop_keyword.strip().lower()
    state_keyword = state_keyword.strip().lower()

    # Synonyms for land.
    land_synonyms = {
        "marginal": ["marginal", "small"],
        "small": ["small", "marginal"],
        "medium": ["medium"],
        "large": ["large"]
    }
    search_terms = set()
    search_terms.add(land_category)
    for synonym in land_synonyms.get(land_category, []):
        search_terms.add(synonym)

    filtered_schemes = []
    for scheme in schemes:
        scheme_text = ' '.join([
            scheme.get('title', ''),
            scheme.get('description', ''),
            scheme.get('benefits', ''),
            scheme.get('eligibility', ''),
        ]).lower()
        
        land_match = (land_category=='all') or any(term in scheme_text for term in search_terms)
        crop_match = (crop_keyword=='all') or (crop_keyword in scheme_text)
        state_match = (state_keyword=='all') or (state_keyword in scheme_text)

        if land_match and crop_match and state_match:
            filtered_schemes.append(scheme)

    if filtered_schemes:
        st.markdown(f"### Found {len(filtered_schemes)} matching schemes:")
        for s in filtered_schemes:
            st.markdown(
                f"#### [{s.get('title', 'Untitled Scheme')}]({s.get('url', '#')})\n"
                f"{s.get('description', '')}\n\n**Benefits:** {s.get('benefits', '')}\n\n"
                f"**Eligibility:** {s.get('eligibility','')}\n\n---"
            )
    else:
        st.info("No matched schemes found. Try broadening your filter criteria.")

def generate_weather_alerts(weather_data, crop_type):
    """Generate weather-based alerts"""
    alerts = []
    
    # Frost alert
    if weather_data['frost_risk'] > 0.7:
        alerts.append({
            'severity': 'high',
            'message': f'Frost warning for {crop_type} - temperature may drop below 2°C'
        })
    
    # Drought alert
    if weather_data['drought_risk'] > 0.6:
        alerts.append({
            'severity': 'medium',
            'message': f'Drought conditions expected - consider water conservation'
        })
    
    # Normal conditions
    if weather_data['frost_risk'] < 0.3 and weather_data['drought_risk'] < 0.3:
        alerts.append({
            'severity': 'low',
            'message': 'Weather conditions favorable for crop growth'
        })
    
    return alerts



def weather_risk_monitor(pipeline):
    st.header("🌤️ Live Weather Risk Monitoring System")

    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🌡️ Active Farmers", "1,247", "↑ 23")
    with col2: st.metric("⚠️ High Risk Alerts", "15", "↓ 3")
    with col3: st.metric("🌧️ Rainfall Alerts", "8", "→ 0")
    with col4: st.metric("✅ Safe Conditions", "1,224", "↑ 20")

    """Weather risk monitoring dashboard"""
    st.markdown("## 🌦️ Weather Risk Monitor")
    
    alert_system = WeatherAlertSystem()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚨 Active Weather Alerts")
        
        if st.button("🔄 Check for New Alerts", type="primary"):
            with st.spinner("Scanning weather conditions..."):
                alerts_generated = alert_system.run_once()
                st.success(f"✅ Scan complete! Generated {alerts_generated} alerts")
        
        # Display recent alerts
        recent_alerts = alert_system.list_recent_alerts(limit=20)
        
        if recent_alerts:
            alerts_df = pd.DataFrame(recent_alerts)
            
            # Format and display
            for i, alert in enumerate(recent_alerts[:5]):
                severity_color = {
                    'high': 'error',
                    'medium': 'warning', 
                    'low': 'info'
                }.get(alert['severity'], 'info')
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        getattr(st, severity_color)(f"**{alert['alert_type'].title()}**: {alert['message']}")
                    with col2:
                        st.write(f"Farmer ID: {alert['farmer_id']}")
                    with col3:
                        st.write(f"Severity: {alert['severity'].upper()}")
        else:
            st.info("No recent alerts. Weather conditions are stable.")
    
    # Fetch real weather data for all cities
    weather_data = []
    alerts_feed = []
    for city in CITIES:
        data = get_weather(city["lat"], city["lon"])
        if data:
            risk_level = min(max(data["main"]["temp"] / 50, 0), 1)  # simple risk proxy
            weather_data.append({
                "lat": city["lat"],
                "lon": city["lon"],
                "city": city["name"],
                "risk_level": risk_level,
                "farmers_count": int(100 + risk_level * 200)
            })
            _, alerts = parse_weather_data(data)
            for alert, severity in alerts:
                alerts_feed.append({"city": city["name"], "alert": alert, "severity": severity})

    # Weather map
    st.subheader("🗺️ Regional Weather Risk Map")
    weather_df = pd.DataFrame(weather_data)
    fig_map = px.scatter_mapbox(
        weather_df, lat="lat", lon="lon", color="risk_level",
        size="farmers_count", hover_name="city",
        color_continuous_scale="RdYlGn_r", size_max=50, zoom=4
    )
    fig_map.update_layout(mapbox_style="open-street-map", height=400, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # # Live weather reports
    # st.markdown("### 📄 Latest Weather Reports")
    # for city in CITIES:
    #     data = get_weather(city["lat"], city["lon"])
    #     if data:
    #         report, _ = parse_weather_data(data)
    #         st.markdown(report)
    display_weather_reports()


    # # Live alerts feed
    # st.subheader("📡 Live Weather Alerts Feed")
    # severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    # for alert in alerts_feed:
    #     st.markdown(f"**{severity_color[alert['severity']]} {alert['alert']}** - {alert['city']}")
    display_alerts(alerts_feed)

def policy_advisor_with_filters(pipeline, land_size, crop_type, state):
    st.header('🏛️ Dynamic Government Policy Advisor')
    st.markdown("""
        **Real-time policy matching engine that connects farmers to relevant 
        government schemes, subsidies, and insurance policies based on their profile and current conditions.**
    """)
    try:
        with open('myschemes_full.json', 'r', encoding='utf-8') as f:
            policies = json.load(f)
    except FileNotFoundError:
        st.error('❌ `myschemes_full.json` not found. Please scrape MyScheme first.')
        return
    
    st.subheader('🔍 Matched Policies Based on Your Profile')
    
    # Filter policies based on passed keywords (simple example assuming policies have keys for those)
    filtered_policies = []
    for policy in policies:
        if (land_size == 'All' or policy.get('land_size', '').lower() == land_size.lower()) and \
           (crop_type == 'All' or crop_type.lower() in policy.get('crops', '').lower()) and \
           (state == 'All' or state.lower() == policy.get('state', '').lower()):
            filtered_policies.append(policy)
    
    if filtered_policies:
        for p in filtered_policies:
            st.markdown(f"### {p.get('name')}\n{p.get('description')}\n")
    else:
        st.info('No policies matched your profile criteria.')


def portfolio_dashboard(pipeline):
    st.header("📊 Real-Time Portfolio Analytics Dashboard")
    
    # Seed if empty, for demo only (remove in production!)
    count = pipeline.conn.execute("SELECT COUNT(*) FROM portfolio_metrics").fetchone()[0]
    if count < 30:
        pipeline.seed_portfolio_history(60)  # Seed 2 months of demo data


    # Initialize data if empty
    if st.button("🔄 Refresh/Initialize Database"):
        pipeline.seed_farmers(200)  # Create 200 farmers
        pipeline.seed_loans_for_farmers()  # Create loans
        pipeline.calculate_and_store_portfolio_metrics()  # Calculate metrics
        st.success("Database initialized with real farmer data!")
    
      # Get current metrics with error handling
    try:
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        # Initialize empty database first
        pipeline.seed_farmers(50)
        pipeline.seed_loans_for_farmers()
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
        st.success("Initialized database with sample data!")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Portfolio", f"₹{current_metrics['total_portfolio']:,.0f}")
    with col2:
        st.metric("👥 Total Farmers", f"{current_metrics['total_farmers']:,}")
    with col3:
        st.metric("📈 Total Loans", f"{current_metrics['total_loans']:,}")
    with col4:
        st.metric("⚠️ Default Rate", f"{current_metrics['default_rate']:.2f}%")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("🎯 Avg Credit Score", f"{current_metrics['avg_credit_score']:.0f}")
    with col6:
        st.metric("🔄 Active Loans", f"{current_metrics['active_loans']:,}")
    
    # Get trends data
    trends_df = pipeline.get_portfolio_trends(30)
    
    if not trends_df.empty:
        # Portfolio value trend
        fig1 = px.line(trends_df, x='date', y='total_portfolio_value', 
                      title='Portfolio Value Trend (30 Days)')
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Default rate trend
            fig2 = px.line(trends_df, x='date', y='default_rate', 
                          title='Default Rate Trend')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Credit score trend
            fig3 = px.line(trends_df, x='date', y='avg_credit_score', 
                          title='Average Credit Score Trend')
            st.plotly_chart(fig3, use_container_width=True)
    
    # Real farmer data table
    with st.expander("📋 View Farmer Database"):
        farmers_df = pd.read_sql_query("""
            SELECT f.farmer_id, f.name, f.crop_type, f.land_size,
                   COUNT(l.loan_id) as total_loans,
                   COALESCE(SUM(l.amount), 0) as total_borrowed,
                   COALESCE(AVG(l.credit_score), 0) as avg_credit_score
            FROM farmers f
            LEFT JOIN loans l ON f.farmer_id = l.farmer_id
            GROUP BY f.farmer_id
            ORDER BY total_borrowed DESC
            LIMIT 50
        """, pipeline.conn)
        st.dataframe(farmers_df, use_container_width=True)
    
    # Loan status distribution
    loan_status_df = pd.read_sql_query("""
        SELECT status, COUNT(*) as count, SUM(amount) as total_amount
        FROM loans GROUP BY status
    """, pipeline.conn)
    
    if not loan_status_df.empty:
        fig4 = px.pie(loan_status_df, values='count', names='status',
                     title='Loan Status Distribution')
        st.plotly_chart(fig4, use_container_width=True)

    @st.cache_resource
    def get_carbon_credit_model():
        # Generate demo 'real' data for model
        np.random.seed(42)
        N = 300
        data = {
            "area": np.random.uniform(1, 80, N),
            "ndvi": np.random.uniform(0.2, 0.9, N),
            "soil_carbon": np.random.uniform(5, 40, N),
            "rainfall": np.random.uniform(400, 1800, N),
            "type_afforestation": np.random.binomial(1, 0.24, N),
            "type_nitill": np.random.binomial(1, 0.26, N),
            "type_covercropping": np.random.binomial(1, 0.25, N),
            "type_rice": np.random.binomial(1, 0.25, N),
            "verified": np.random.binomial(1, 0.93, N),
        }
        # FOR DEMO: the true carbon credit is a nonlinear mix of above, plus randomness:
        y = (
            data["area"] * data["ndvi"] * (data["type_afforestation"]*1.35 + data["type_nitill"]*1.09 +
                data["type_covercropping"]*1.13 + data["type_rice"]*1.00)
            * data["verified"] * 0.95
            + 0.0015 * data["rainfall"]
            - 0.21 * data["soil_carbon"]
            + np.random.normal(0, 1.5, N)
        )
        X = pd.DataFrame(data)
        X["type_afforestation"] = data["type_afforestation"]
        X["type_nitill"] = data["type_nitill"]
        X["type_covercropping"] = data["type_covercropping"]
        X["type_rice"] = data["type_rice"]
        y = np.maximum(y, 0) # can't have negative credits

        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(X, y)
        return model

    ml_model = get_carbon_credit_model()

    # --- Project Input Form (& ML prediction) ---
    TYPE_MAP = {
        "Afforestation": [1, 0, 0, 0],
        "No-till": [0, 1, 0, 0],
        "Cover Cropping": [0, 0, 1, 0],
        "Rice": [0, 0, 0, 1]
    }

    st.markdown("#### 💡 Estimate/Certify New Carbon Credits (powered by ML)")
    with st.form("carbon_ml"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_area = st.number_input("Area (ha)", 0.1, 300.0, value=6.0)
            in_type = st.selectbox("Project Type", list(TYPE_MAP))
        with col2:
            in_ndvi = st.slider("Avg NDVI (satellite)", 0.15, 0.95, 0.6)
            in_soil = st.number_input("Baseline Soil Carbon (t/ha)", 1.0, 80.0, value=14.0)
        with col3:
            in_rain = st.number_input("Rainfall (mm/yr)", 300, 2200, value=900)
            in_verified = st.checkbox("Practices Verified", value=True)

        in_location = st.text_input("Farm Location", "Unknown")  # <-- moved here
        ml_submit = st.form_submit_button("Estimate Credits")

    # --- Perform ML prediction ---
    pred_credit = None
    if ml_submit:
        in_feats = np.array([
            in_area, in_ndvi, in_soil, in_rain,
            *TYPE_MAP[in_type], int(in_verified)
        ]).reshape(1, -1)
        pred_credit = ml_model.predict(in_feats)[0]
        pred_credit = max(0, round(float(pred_credit), 2))
        st.success(f"ML-estimated Carbon Credits: **{pred_credit} tCO₂e** (for this project)")

        # Save to portfolio
        if "cc_portfolio" not in st.session_state:
            st.session_state["cc_portfolio"] = []
        add_row = {
            "Project": f"User Project {len(st.session_state['cc_portfolio'])+1}",
            "Type": in_type,
            "Area (ha)": in_area,
            "NDVI": in_ndvi,
            "SoilC (t/ha)": in_soil,
            "Rain (mm)": in_rain,
            "Verified": in_verified,
            "ML Credits (tCO₂e)": pred_credit
        }
        st.session_state["cc_portfolio"].append(add_row)

        # --- NEW: Store in database ledger ---
        farm_id = f"FARM{len(st.session_state['cc_portfolio'])}"
        location = st.text_input("Enter Farm Location", "Unknown")
        store_credit_transaction(farm_id, location, "Verified" if in_verified else "Unverified", pred_credit)


    # --- Portfolio Display ---
    st.markdown("#### Portfolio Carbon Credits (from session)")
    if "cc_portfolio" in st.session_state and st.session_state["cc_portfolio"]:
        cdf = pd.DataFrame(st.session_state["cc_portfolio"])
        total_credits = cdf['ML Credits (tCO₂e)'].sum()
        market_value_inr = total_credits * CREDIT_PRICE_USD * USD_TO_INR
        roi = market_value_inr * 0.25  # example 25% margin
        cars_equiv = total_credits / CAR_EQUIV_TON
        trees_equiv = total_credits / TREE_EQUIV_TON

        st.dataframe(cdf, use_container_width=True)
        st.metric("Total ML-estimated Credits", f"{total_credits:.2f} tCO₂e")
        st.metric("Estimated Market Value (₹)", f"{market_value_inr:,.0f}")
        st.metric("Projected ROI (₹)", f"{roi:,.0f}")
        st.metric("Cars Off Road (equivalent)", f"{cars_equiv:,.0f}")
        st.metric("Trees Planted (equivalent)", f"{trees_equiv:,.0f}")
        st.bar_chart(cdf.set_index("Project")["ML Credits (tCO₂e)"])
        st.download_button("Download Portfolio (CSV)", cdf.to_csv(index=False), file_name="carbon_portfolio.csv")
    else:
        st.info("No carbon credits in portfolio yet. Use the form above to add projects!")

    # --- Carbon Credit Ledger (from DB) ---
    st.markdown("#### 📜 Blockchain Ledger (Verified Records)")

    df = load_data()

    if df.empty:
        st.info("Ledger is empty. Add projects above or seed mock data.")
    else:
        # Show ledger table
        st.dataframe(df, use_container_width=True)

        # Show blockchain hashes
# Tamper-Evidence

# If anyone tries to alter even one record (say, inflating a farmer’s credits), the hash changes.

# Since the next block references the old hash, the chain breaks — making fraud or manipulation easily detectable.

# Transparency & Trust

# Farmers, buyers, and regulators can trust the carbon credit ledger because it’s cryptographically verifiable, not just a normal database entry.

# Auditability

# Regulators or verifiers can check the hash chain integrity instead of relying only on raw SQL records.

# This reduces the chance of disputes.

# “Blockchain without Blockchain”

# You’re not running a heavy blockchain node or smart contracts.

# You’re creating a lightweight, blockchain-style audit trail inside SQLite — faster, cheaper, and perfect for a prototype.

# Future-Ready

# If AgriCred scales, you could migrate these records to a real blockchain (like Polygon or Hyperledger).

# Since you already have hashes, migration will be straightforward.
        with st.expander("🔗 Blockchain Hash Verification"):
            for idx, row in df.iterrows():
                st.markdown(f"**Block {row['id']}** | Farm: {row['farm_id']} | Status: {row['verification_status']}")
                st.code(f"Hash: {row['hash']}\nPrev: {row['prev_hash']}", language="bash")


def market_intelligence_dashboard():
    """Market intelligence and commodity analysis"""
    st.markdown("## 💹 Market Intelligence & Commodity Analysis")
    
    # Market overview
    st.markdown("### 📊 Agricultural Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        wheat_price = random.uniform(2000, 2500)
        st.metric("Wheat Price", f"₹{wheat_price:.0f}/qt", "+4.2%")
    
    with col2:
        rice_price = random.uniform(2200, 2800)
        st.metric("Rice Price", f"₹{rice_price:.0f}/qt", "-1.8%")
    
    with col3:
        cotton_price = random.uniform(4500, 5500)
        st.metric("Cotton Price", f"₹{cotton_price:.0f}/qt", "+8.7%")
    
    with col4:
        soybean_price = random.uniform(3200, 4000)
        st.metric("Soybean Price", f"₹{soybean_price:.0f}/qt", "+2.1%")
    
    # Price trends
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 6-Month Price Trends")
        
        # Generate price data
        dates = pd.date_range(start='2025-03-01', end='2025-08-31', freq='D')
        price_data = pd.DataFrame({
            'Date': dates,
            'Wheat': np.cumsum(np.random.normal(0, 15, len(dates))) + 2200,
            'Rice': np.cumsum(np.random.normal(0, 12, len(dates))) + 2400,
            'Cotton': np.cumsum(np.random.normal(0, 25, len(dates))) + 5000
        })
        
        fig_prices = px.line(
            price_data,
            x='Date',
            y=['Wheat', 'Rice', 'Cotton'],
            title='Commodity Price Movements',
            labels={'value': 'Price (₹/quintal)', 'variable': 'Commodity'}
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Global Market Impact")
        
        # Global factors
        global_factors = {
            'Factor': ['Export Demand', 'International Prices', 'Currency Impact', 'Supply Chain', 'Weather Events'],
            'Impact Score': [random.uniform(0.6, 0.9) for _ in range(5)],
            'Trend': ['↑ Positive', '↓ Negative', '→ Stable', '↑ Positive', '↓ Negative']
        }
        
        df_global = pd.DataFrame(global_factors)
        
        fig_global = px.bar(
            df_global,
            x='Factor',
            y='Impact Score',
            color='Impact Score',
            title='Global Market Factors Impact',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_global, use_container_width=True)
    
    # Market insights for lenders
    st.markdown("---")
    st.subheader("💡 Lending Strategy Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="financier-insight">
        <h4>🎯 High Opportunity Crops</h4>
        <ul>
        <li><strong>Cotton:</strong> Strong export demand (+8.7%)</li>
        <li><strong>Wheat:</strong> Government procurement support</li>
        <li><strong>Organic Produce:</strong> Premium pricing trend</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="financier-insight">
        <h4>⚠️ Risk Segments</h4>
        <ul>
        <li><strong>Rice:</strong> Price volatility (-1.8%)</li>
        <li><strong>Sugarcane:</strong> Processing delays</li>
        <li><strong>Pulses:</strong> Import competition</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="financier-insight">
        <h4>📈 Portfolio Recommendations</h4>
        <ul>
        <li><strong>Increase:</strong> Cotton loan exposure</li>
        <li><strong>Maintain:</strong> Wheat portfolio balance</li>
        <li><strong>Monitor:</strong> Rice segment closely</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def policy_advisor(pipeline):
    st.header("🏛️ Dynamic Government Policy Advisor")
    
    st.markdown("""
    **Real-time policy matching engine that connects farmers to relevant government schemes, 
    subsidies, and insurance policies based on their profile and current conditions.**
    """)

    # Load scraped scheme data
    try:
        with open("myschemes_full.json", "r", encoding="utf-8") as f:
            policies = json.load(f)
    except FileNotFoundError:
        st.error("❌ `myschemes_full.json` not found. Please scrape MyScheme first.")
        return

    # --- User Filters ---
    st.subheader("🔍 Find Relevant Policies")
    
    col1, col2 = st.columns(2)
    with col1:
        search_land_size = st.selectbox(
            "Land Size Category", 
            ["All", "Marginal (<1 acre)", "Small (1-2 acres)", "Medium (2-10 acres)", "Large (>10 acres)"]
        )
        search_crop = st.selectbox(
            "Crop Type", 
            ["All", "Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Oilseeds"]
        )
    
    with col2:
        search_state = st.selectbox(
            "State", 
            ["All", "Uttar Pradesh", "Maharashtra", "Punjab", "Haryana", "Bihar"]
        )
        search_category = st.selectbox(
            "Policy Category",
            ["All", "Credit Schemes", "Insurance", "Subsidies", "Market Support"]
        )
    
    if st.button("🔍 Search Policies", use_container_width=True):

        def match_score(policy):
            """Calculate matching score based on user filters."""
            score = 0
            text_blob = f"{policy.get('title', '')} {policy.get('benefits', '')} {policy.get('eligibility', '')}".lower()
            
            # Land size matching
            if search_land_size != "All" and re.search(search_land_size.split()[0].lower(), text_blob):
                score += 0.25
            
            # Crop matching
            if search_crop != "All" and search_crop.lower() in text_blob:
                score += 0.25

            # State matching
            if search_state != "All" and search_state.lower() in text_blob:
                score += 0.25

            # Category (basic keyword-based)
            if search_category != "All" and search_category.lower() in text_blob:
                score += 0.25

            return score

        # Compute scores for all policies
        for policy in policies:
            policy["score"] = match_score(policy)
        
        # Sort by score, highest first
        matched_policies = sorted(policies, key=lambda x: x["score"], reverse=True)
        
        # Filter out very low matches
        matched_policies = [p for p in matched_policies if p["score"] > 0]

        # Display
        st.subheader(f"📋 Found {len(matched_policies)} Matching Policies")
        
        if not matched_policies:
            st.warning("No exact matches found. Try selecting broader filters.")
        else:
            for i, policy in enumerate(matched_policies):
                with st.expander(f"📄 {policy.get('title', 'Unnamed Scheme')}", expanded=i == 0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Benefits:** {policy.get('benefits', 'N/A')}")
                        st.write(f"**Eligibility:** {policy.get('eligibility', 'N/A')}")
                        st.write(f"**URL:** {policy.get('url', 'N/A')}")
                    
                    with col2:
                        st.progress(policy["score"])
                        st.caption(f"Eligibility Match: {policy['score'] * 100:.0f}%")



# --------- Caching ----------
@st.cache_data(show_spinner=False, ttl=600)
def fetch_current_weather_by_coords(lat: float, lon: float):
    if not WEATHER_API_KEY:
        raise RuntimeError("⚠️ WEATHER_API_KEY not set in .env/config.py")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def fetch_current_weather_by_city(city: str):
    if city in CITY_COORDS:
        return fetch_current_weather_by_coords(*CITY_COORDS[city])
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": WEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        raise ValueError(f"City not found: {city}")
    return fetch_current_weather_by_coords(arr[0]["lat"], arr[0]["lon"])

def format_weather_human(data: dict) -> str:
    name = data.get("name") or f"{data['coord']['lat']:.3f},{data['coord']['lon']:.3f}"
    w = data.get("weather", [{}])[0]
    desc = (w.get("description") or "").title()
    main = data.get("main", {})
    wind = data.get("wind", {})
    return f"**{name}** — {desc}  |  🌡 {main.get('temp')}°C  •  💧 {main.get('humidity')}%  •  💨 {wind.get('speed')} m/s"

# --------- Agmarknet / Mandi prices ----------
@st.cache_data(show_spinner=False, ttl=900)
def fetch_mandi_prices(limit=2000, state=None, commodity=None):
    if not MARKET_API_KEY:
        raise RuntimeError("⚠️ MARKET_API_KEY not set in .env/config.py")
    params = {
        "api-key": MARKET_API_KEY,
        "format": "json",
        "limit": str(limit),
    }
    if state:
        params["filters[state]"] = state
    if commodity:
        params["filters[commodity]"] = commodity
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("records", []))
    for col in ("min_price", "max_price", "modal_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def executive_summary_dashboard():
    """Executive summary dashboard for senior management"""
    st.markdown("## 🏠 Executive Summary - Agricultural Portfolio Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = st.session_state.pipeline.calculate_and_store_portfolio_metrics()

    with col1:
        st.metric(
            "Portfolio Value",
            f"₹{metrics['total_portfolio']/1e7:.1f}Cr",
            f"{metrics.get('portfolio_value_growth','+12.7%')}",  # Add logic for growth if needed
            help="Total agricultural loan portfolio value"
        )

    with col2:
        st.metric(
            "Active Farmers",
            f"{metrics['total_farmers']:,}",
            f"+{metrics.get('new_farmers','59')}",
            help="Number of farmers with active loans"
        )

    with col3:
        st.metric(
            "Default Rate",
            f"{metrics['default_rate']:.1f}%",
            "-1.8%",
            help="Current portfolio default rate (industry avg: 6.1%)"
        )

    with col4:
        st.metric(
            "Avg Credit Score",
            f"{int(metrics['avg_credit_score'])}",
            f"+{metrics.get('credit_score_change',21)}",
            help="Average credit score of portfolio"
        )

    with col5:
        st.metric(
            "Risk-Adjusted ROI",
            "14.7%",
            "+2.1%",
            help="Risk-adjusted return on investment"
        )
    # Portfolio composition
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Portfolio Composition by Crop Type")
        
        # Generate realistic portfolio data
        crop_data = {
            'Crop': ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Soybean', 'Maize', 'Others'],
            'Portfolio Value (₹Cr)': [187.5, 164.2, 142.8, 98.7, 86.3, 74.5, 93.3],
            'Farmers Count': [6247, 5832, 4156, 2897, 3247, 2854, 3223],
            'Avg Loan Size (₹L)': [3.2, 2.8, 4.1, 3.6, 2.7, 2.4, 2.9],
            'Default Rate (%)': [3.2, 4.1, 5.8, 3.9, 4.7, 4.2, 5.1]
        }
        
        df_crops = pd.DataFrame(crop_data)
        
        # Portfolio composition pie chart
        fig_pie = px.pie(
            df_crops, 
            values='Portfolio Value (₹Cr)', 
            names='Crop',
            title="Portfolio Distribution by Crop Value",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        risk_data = {
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [18247, 8456, 1753],
            'Portfolio %': [64.1, 29.7, 6.2]
        }
        
        df_risk = pd.DataFrame(risk_data)
        
        fig_risk = px.bar(
            df_risk,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Low Risk': '#28a745',
                'Medium Risk': '#ffc107', 
                'High Risk': '#dc3545'
            },
            title="Farmers by Risk Category"
        )
        fig_risk.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Geographic performance
    st.markdown("---")
    st.subheader("🗺️ Geographic Performance Overview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # State-wise performance data
        state_data = {
            'State': ['Punjab', 'UP', 'Maharashtra', 'Karnataka', 'AP', 'Gujarat', 'MP', 'WB'],
            'Portfolio (₹Cr)': [156.2, 134.7, 128.4, 98.6, 89.3, 76.8, 92.1, 71.2],
            'Farmers': [4256, 5847, 3654, 2987, 3156, 2245, 3847, 2464],
            'Default Rate': [2.8, 5.2, 4.1, 3.6, 4.8, 3.2, 5.7, 6.1],
            'Avg Loan Size': [3.67, 2.31, 3.51, 3.30, 2.83, 3.42, 2.40, 2.89]
        }
        
        df_states = pd.DataFrame(state_data)
        
        fig_geo = px.scatter(
            df_states,
            x='Portfolio (₹Cr)',
            y='Default Rate',
            size='Farmers',
            color='State',
            title="Portfolio Performance: Size vs Risk by State",
            hover_data=['Avg Loan Size']
        )
        fig_geo.update_layout(height=500)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Key Insights")
        
        st.markdown("""
        <div class="financier-insight">
        <h4>🎯 Strategic Opportunities</h4>
        <ul>
        <li><strong>Punjab Portfolio:</strong> Lowest default rate (2.8%) - expand operations</li>
        <li><strong>Cotton Segment:</strong> High margins but elevated risk - enhance screening</li>
        <li><strong>Technology Adoption:</strong> 23% boost in repayment rates for tech-enabled farmers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="financier-insight">
        <h4>⚠️ Risk Alerts</h4>
        <ul>
        <li><strong>West Bengal:</strong> Default rate trending upward (6.1%)</li>
        <li><strong>Monsoon Impact:</strong> 847 farmers in high-risk weather zones</li>
        <li><strong>Market Volatility:</strong> Cotton prices down 12% this quarter</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("---")
    st.subheader("📈 12-Month Performance Trends")
    
    # Generate trend data
    months = pd.date_range(start='2024-09-01', end='2025-08-31', freq='MS')
    trend_data = {
        'Month': months,
        'Portfolio Value': np.random.normal(75, 5, 12).cumsum() + 700,
        'Default Rate': np.random.normal(0, 0.3, 12).cumsum() + 5.5,
        'New Loans': np.random.poisson(450, 12),
        'ROI': np.random.normal(0, 0.5, 12).cumsum() + 13
    }
    
    df_trends = pd.DataFrame(trend_data)
    
    # Create subplots
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Portfolio Growth', 'Default Rate Trend', 'Monthly New Loans', 'ROI Trend'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Portfolio growth
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['Portfolio Value'], 
                  name='Portfolio (₹Cr)', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Default rate
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['Default Rate'], 
                  name='Default Rate (%)', line=dict(color='#ff7f0e')),
        row=1, col=2
    )
    
    # New loans
    fig_trends.add_trace(
        go.Bar(x=df_trends['Month'], y=df_trends['New Loans'], 
               name='New Loans', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # ROI
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['ROI'], 
                  name='ROI (%)', line=dict(color='#d62728')),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)



def system_configuration():
    """System configuration and settings"""
    st.markdown("## ⚙️ System Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Platform Settings")
        
        with st.form("system_config"):
            st.markdown("#### Risk Assessment Parameters")
            default_threshold = st.slider("Default Risk Threshold", 0.0, 1.0, 0.3)
            weather_weight = st.slider("Weather Risk Weight", 0.0, 1.0, 0.25)
            market_weight = st.slider("Market Risk Weight", 0.0, 1.0, 0.20)
            
            st.markdown("#### Alert Settings")
            alert_frequency = st.selectbox("Alert Check Frequency", ["Hourly", "Daily", "Weekly"])
            email_alerts = st.checkbox("Email Notifications", True)
            sms_alerts = st.checkbox("SMS Alerts", False)
            
            st.markdown("#### Data Refresh")
            auto_refresh = st.checkbox("Auto Refresh Data", True)
            refresh_interval = st.selectbox("Refresh Interval", ["15 min", "30 min", "1 hour", "2 hours"])
            
            if st.form_submit_button("💾 Save Configuration"):
                st.success("✅ Configuration saved successfully!")
    
    with col2:
        st.subheader("📊 System Status")
        
        st.metric("System Health", "99.7%", "All systems operational")
        st.metric("API Response Time", "234ms", "Excellent")
        st.metric("Data Accuracy", "98.9%", "High quality")
        st.metric("Last Updated", "2 min ago", "Real-time")
        
        st.markdown("---")
        st.subheader("🔗 API Connections")
        
        st.success("✅ Weather API - Connected")
        st.success("✅ Market Data API - Connected") 
        st.success("✅ Credit Bureau API - Connected")
        st.warning("⚠️ Satellite API - Limited")
        
        st.markdown("---")
        st.subheader("📁 Data Sources")
        
        st.info("🏦 Internal Database: 847,234 records")
        st.info("🌦️ Weather Data: 1 sources")
        st.info("💹 Market Data: stimulated data")
        st.info("🛰️ Satellite Data: coming soon")



def portfolio_analytics_dashboard():
    """Detailed portfolio analytics for loan officers"""
    st.markdown("## 📊 Portfolio Analytics - Deep Dive")
    
    pipeline = st.session_state.pipeline
    
    # Ensure we have data
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Refresh Portfolio Data", type="primary"):
            with st.spinner("Updating portfolio metrics..."):
                # Seed farmers if needed
                farmer_count = pipeline.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
                if farmer_count < 200:
                    pipeline.seed_farmers(500)
                    pipeline.seed_loans_for_farmers()
                
                # Calculate and store metrics
                pipeline.seed_portfolio_history(90)  # 3 months of history
                st.success("✅ Portfolio data refreshed!")
    
    with col2:
        st.metric("Data Freshness", "Live", help="Real-time portfolio data")
    with col3:
        st.metric("Coverage", "99.7%", help="Data coverage across portfolio")
    
    # Portfolio overview metrics
    try:
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
        
        st.markdown("### 📈 Current Portfolio Snapshot")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Farmers",
                f"{current_metrics['total_farmers']:,}",
                help="Active farmers in portfolio"
            )
        
        with col2:
            st.metric(
                "Active Loans", 
                f"{current_metrics['total_loans']:,}",
                help="Number of active loans"
            )
        
        with col3:
            st.metric(
                "Portfolio Value",
                f"₹{current_metrics['total_portfolio']/10000000:.1f}Cr",
                help="Total outstanding loan amount"
            )
        
        with col4:
            st.metric(
                "Default Rate",
                f"{current_metrics['default_rate']:.1f}%",
                help="Current portfolio default rate"
            )
        
        with col5:
            st.metric(
                "Avg Credit Score",
                f"{int(current_metrics['avg_credit_score'])}",
                help="Average credit score of borrowers"
            )
        
        # Portfolio trends
        st.markdown("---")
        st.markdown("### 📊 Portfolio Performance Trends")
        
        trend_data = pipeline.get_portfolio_trends(60)  # 60 days
        if not trend_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio value trend
                fig_portfolio = px.line(
                    trend_data,
                    x='date',
                    y='total_portfolio_value',
                    title='Portfolio Value Growth',
                    labels={'total_portfolio_value': 'Portfolio Value (₹)', 'date': 'Date'}
                )
                fig_portfolio.update_traces(line_color='#1f77b4', line_width=3)
                st.plotly_chart(fig_portfolio, use_container_width=True)
            
            with col2:
                # Default rate trend
                fig_default = px.line(
                    trend_data,
                    x='date',
                    y='default_rate',
                    title='Default Rate Trend',
                    labels={'default_rate': 'Default Rate (%)', 'date': 'Date'}
                )
                fig_default.update_traces(line_color='#ff7f0e', line_width=3)
                st.plotly_chart(fig_default, use_container_width=True)
            
            # Credit score distribution
            fig_credit = px.line(
                trend_data,
                x='date',
                y='avg_credit_score',
                title='Average Credit Score Trend',
                labels={'avg_credit_score': 'Avg Credit Score', 'date': 'Date'}
            )
            fig_credit.update_traces(line_color='#2ca02c', line_width=3)
            st.plotly_chart(fig_credit, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading portfolio metrics: {str(e)}")
        st.info("Please refresh the portfolio data to generate metrics.")






#==========================================
def main():
    """Main application function"""
    
    # Display header
    display_main_header()
    
    # Sidebar navigation
    page = display_sidebar()

     # Initialize components
    pipeline = initialize_data_pipeline()
    model, scaler = load_models()
    
    if model is None:
        st.error("⚠️ Models not found. Please run advanced_ml_model.py first to train the models.")
        return
    # Initialize database with farmers on first run
    farmer_count = pipeline.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
    if farmer_count == 0:
        st.info("Initializing database with farmer data...")
        pipeline.seed_farmers(2000)
        pipeline.seed_loans_for_farmers()
        pipeline.calculate_and_store_portfolio_metrics()
    
    if page == "🏠 Executive Summary":
        executive_summary_dashboard()
    elif page == "📊 Portfolio Analytics":
        portfolio_analytics_dashboard()
    elif page == "🎯 Credit Risk Scoring":
        credit_risk_scoring_dashboard()
    elif page == "🤖 Agentic AI Intelligence":
        agentic_ai_demo()
    elif page == "🌦️ Weather Risk Monitor":
        weather_risk_monitor(pipeline)
    elif page == "💹 Market Intelligence":
        market_intelligence_dashboard()
    elif page == "📈 Performance Analytics":
        performance_analytics()
    elif page == "⚙️ System Configuration":
        system_configuration()

if __name__ == "__main__":
    main()


