#!/usr/bin/env python3
"""
Setup script for AgriCred AI platform
Initializes database, trains models, and sets up the system
"""

import os
import sys
import sqlite3
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 'xgboost', 
        'lightgbm', 'shap', 'joblib', 'plotly', 'folium', 'requests', 
        'geopy', 'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def setup_database():
    """Initialize the SQLite database with required tables"""
    print("\n🗄️ Setting up database...")
    
    try:
        conn = sqlite3.connect('agricred_data.db')
        
        # Create tables
        tables = {
            'farmers': '''
                CREATE TABLE IF NOT EXISTS farmers (
                    farmer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    land_size REAL NOT NULL,
                    crop_type TEXT NOT NULL,
                    soil_type TEXT,
                    phone_number TEXT,
                    registration_date DATE DEFAULT CURRENT_DATE
                )
            ''',
            'weather_data': '''
                CREATE TABLE IF NOT EXISTS weather_data (
                    id INTEGER PRIMARY KEY,
                    farmer_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL,
                    humidity REAL,
                    rainfall REAL,
                    wind_speed REAL,
                    pressure REAL,
                    weather_condition TEXT,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            ''',
            'market_prices': '''
                CREATE TABLE IF NOT EXISTS market_prices (
                    id INTEGER PRIMARY KEY,
                    crop_type TEXT NOT NULL,
                    mandi_name TEXT,
                    price_per_quintal REAL NOT NULL,
                    date DATE DEFAULT CURRENT_DATE,
                    state TEXT,
                    district TEXT
                )
            ''',
            'soil_health': '''
                CREATE TABLE IF NOT EXISTS soil_health (
                    farmer_id INTEGER PRIMARY KEY,
                    ph_level REAL,
                    nitrogen REAL,
                    phosphorus REAL,
                    potassium REAL,
                    organic_carbon REAL,
                    last_updated DATE DEFAULT CURRENT_DATE,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            ''',
            'loan_history': '''
                CREATE TABLE IF NOT EXISTS loan_history (
                    loan_id INTEGER PRIMARY KEY,
                    farmer_id INTEGER,
                    amount REAL NOT NULL,
                    duration_months INTEGER NOT NULL,
                    interest_rate REAL NOT NULL,
                    status TEXT DEFAULT 'Active',
                    disbursed_date DATE DEFAULT CURRENT_DATE,
                    due_date DATE,
                    repaid_amount REAL DEFAULT 0,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            ''',
            'government_schemes': '''
                CREATE TABLE IF NOT EXISTS government_schemes (
                    scheme_id INTEGER PRIMARY KEY,
                    scheme_name TEXT NOT NULL,
                    description TEXT,
                    eligibility_criteria TEXT,
                    max_amount REAL,
                    interest_rate REAL,
                    duration_months INTEGER,
                    state TEXT,
                    crop_types TEXT,
                    land_size_min REAL,
                    land_size_max REAL
                )
            ''',
            'weather_alerts': '''
                CREATE TABLE IF NOT EXISTS weather_alerts (
                    id INTEGER PRIMARY KEY,
                    farmer_id INTEGER,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    recommended_action TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT 0,
                    FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
                )
            '''
        }
        
        for table_name, table_sql in tables.items():
            conn.execute(table_sql)
            print(f"✅ Created table: {table_name}")
        
        # Insert sample data
        insert_sample_data(conn)
        
        conn.commit()
        conn.close()
        print("✅ Database setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def insert_sample_data(conn):
    """Insert sample data for testing"""
    print("📊 Inserting sample data...")
    
    # Sample farmers
    sample_farmers = [
        (1, 'Ravi Kumar', 28.6139, 77.2090, 2.5, 'Rice', 'Clay Loam', '+91-9876543210'),
        (2, 'Priya Sharma', 19.0760, 72.8777, 1.8, 'Wheat', 'Sandy Loam', '+91-9876543211'),
        (3, 'Amit Patel', 13.0827, 80.2707, 3.2, 'Cotton', 'Red Soil', '+91-9876543212'),
        (4, 'Sunita Devi', 22.5726, 88.3639, 1.5, 'Sugarcane', 'Alluvial', '+91-9876543213'),
        (5, 'Rajesh Singh', 26.9124, 75.7873, 4.0, 'Soybean', 'Black Soil', '+91-9876543214')
    ]
    
    conn.executemany('''
        INSERT OR REPLACE INTO farmers 
        (farmer_id, name, latitude, longitude, land_size, crop_type, soil_type, phone_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_farmers)
    
    # Sample government schemes
    sample_schemes = [
        (1, 'PM-KISAN', 'Direct income support for farmers', 'Land size ≤ 2 acres', 6000, 0, 12, 'All India', 'All crops', 0, 2),
        (2, 'Pradhan Mantri Fasal Bima Yojana', 'Crop insurance scheme', 'All farmers with cultivable land', 100000, 2, 12, 'All India', 'All crops', 0, 100),
        (3, 'Kisan Credit Card', 'Credit card for agricultural inputs', 'All farmers with land records', 500000, 7, 12, 'All India', 'All crops', 0, 100)
    ]
    
    conn.executemany('''
        INSERT OR REPLACE INTO government_schemes 
        (scheme_id, scheme_name, description, eligibility_criteria, max_amount, interest_rate, duration_months, state, crop_types, land_size_min, land_size_max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_schemes)
    
    print("✅ Sample data inserted!")

def train_models():
    """Train the machine learning models"""
    print("\n🤖 Training machine learning models...")
    
    try:
        # Import and run the ML model training
        from advanced_ml_model import AdvancedCreditModel
        
        model = AdvancedCreditModel()
        
        print("Creating dataset...")
        df = model.create_advanced_dataset(5000)
        
        print("Training ensemble model...")
        X_test, y_test, feature_columns = model.train_ensemble_model(df)
        
        print("✅ Model training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print("\n📝 Creating .env file...")
        
        env_content = """# AgriCred AI Environment Configuration

# API Keys
WEATHER_API_KEY=2b3f5d5c338d963f8eb5857979e2a969
MARKET_API_KEY=
SOIL_HEALTH_API_KEY=

# Database Configuration
DATABASE_PATH=agricred_data.db
MODEL_PATH=advanced_credit_model.pkl
SCALER_PATH=feature_scaler.pkl

# Alert System Configuration
SMS_ENABLED=False
EMAIL_ENABLED=False

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=agricred.log

# Development Configuration
DEBUG=False
TESTING=False
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created!")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 AgriCred AI Platform Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    # Train models
    if not train_models():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run advanced_app.py")
    print("\nTo start the weather alert system:")
    print("python weather_alert_system.py")

if __name__ == "__main__":
    main()
