"""
Configuration file for AgriCred AI platform
Contains API keys, database settings, and system configurations
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
MARKET_API_KEY = os.getenv('MARKET_API_KEY', '')
SOIL_HEALTH_API_KEY = os.getenv('SOIL_HEALTH_API_KEY', '')
VOSK_MODEL_PATH = "vosk_model"  # optional for offline STT
# Database Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', 'agricred_data.db')
MODEL_PATH = os.getenv('MODEL_PATH', 'advanced_credit_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'feature_scaler.pkl')

# Weather API Configuration
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5"
WEATHER_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"

WEATHER_UNITS = "metric"
WEATHER_LANG = "en"

# Market Data Configuration
MARKET_API_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
MARKET_UPDATE_INTERVAL = 3600  # 1 hour in seconds

# Alert System Configuration
ALERT_CHECK_INTERVAL = 7200  # 2 hours in seconds
SMS_ENABLED = os.getenv('SMS_ENABLED', 'False').lower() == 'true'
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'

# Email Configuration (for alerts)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

# SMS Configuration (for alerts)
SMS_PROVIDER = os.getenv('SMS_PROVIDER', 'twilio')
SMS_ACCOUNT_SID = os.getenv('SMS_ACCOUNT_SID', '')
SMS_AUTH_TOKEN = os.getenv('SMS_AUTH_TOKEN', '')
SMS_FROM_NUMBER = os.getenv('SMS_FROM_NUMBER', '')

# Model Configuration
MODEL_UPDATE_INTERVAL = 86400  # 24 hours in seconds
FEATURE_COUNT = 50
DEFAULT_SAMPLE_SIZE = 5000

# Risk Thresholds
FROST_RISK_THRESHOLD = 0.7
DROUGHT_RISK_THRESHOLD = 0.6
FLOOD_RISK_THRESHOLD = 0.5
DISEASE_RISK_THRESHOLD = 0.6

# Credit Scoring Configuration
MIN_CREDIT_SCORE = 150
MAX_CREDIT_SCORE = 1000
DEFAULT_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6

# Geographic Configuration
DEFAULT_LATITUDE = 28.6139  # Delhi
DEFAULT_LONGITUDE = 77.2090
EARTH_RADIUS_KM = 6371

# Crop Configuration
CROP_TYPES = {
    'Rice': {'optimal_temp': (20, 35), 'frost_sensitive': True},
    'Wheat': {'optimal_temp': (15, 25), 'frost_sensitive': False},
    'Cotton': {'optimal_temp': (20, 35), 'frost_sensitive': True},
    'Sugarcane': {'optimal_temp': (20, 35), 'frost_sensitive': True},
    'Soybean': {'optimal_temp': (20, 30), 'frost_sensitive': True},
    'Maize': {'optimal_temp': (18, 32), 'frost_sensitive': True}
}

# Government Schemes Configuration
GOVERNMENT_SCHEMES = {
    'PM-KISAN': {
        'max_amount': 6000,
        'interest_rate': 0,
        'duration': 12,
        'eligibility': 'land_size <= 2'
    },
    'Crop_Insurance': {
        'max_amount': 100000,
        'interest_rate': 2,
        'duration': 12,
        'eligibility': 'all_farmers'
    },
    'Kisan_Credit_Card': {
        'max_amount': 500000,
        'interest_rate': 7,
        'duration': 12,
        'eligibility': 'land_owner'
    }
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'agricred.log')

# Development Configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
TESTING = os.getenv('TESTING', 'False').lower() == 'true'

# Model and Scaler Paths
MODEL_PATH = os.getenv('MODEL_PATH', 'advanced_credit_model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'feature_scaler.pkl')

# Cache Configuration
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour

# Rate Limiting
RATE_LIMIT_ENABLED = True
RATE_LIMIT_CALLS = 1000  # calls per hour
RATE_LIMIT_PERIOD = 3600  # 1 hour in seconds
