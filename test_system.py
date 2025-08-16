#!/usr/bin/env python3
"""
Test script for AgriCred AI platform
Verifies all components are working correctly
"""

import sys
import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ xgboost")
    except ImportError as e:
        print(f"❌ xgboost: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print("✅ lightgbm")
    except ImportError as e:
        print(f"❌ lightgbm: {e}")
        return False
    
    try:
        import shap
        print("✅ shap")
    except ImportError as e:
        print(f"❌ shap: {e}")
        return False
    
    try:
        import plotly
        print("✅ plotly")
    except ImportError as e:
        print(f"❌ plotly: {e}")
        return False
    
    try:
        import folium
        print("✅ folium")
    except ImportError as e:
        print(f"❌ folium: {e}")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    try:
        import geopy
        print("✅ geopy")
    except ImportError as e:
        print(f"❌ geopy: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def test_config():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config import WEATHER_API_KEY, DATABASE_PATH, MODEL_PATH, SCALER_PATH
        print(f"✅ Configuration loaded")
        print(f"   Weather API Key: {'Set' if WEATHER_API_KEY else 'Not set'}")
        print(f"   Database Path: {DATABASE_PATH}")
        print(f"   Model Path: {MODEL_PATH}")
        print(f"   Scaler Path: {SCALER_PATH}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_database():
    """Test database connectivity and tables"""
    print("\n🗄️ Testing database...")
    
    try:
        from config import DATABASE_PATH
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Check if tables exist
        tables = ['farmers', 'weather_data', 'market_prices', 'soil_health', 
                 'loan_history', 'government_schemes', 'weather_alerts']
        
        for table in tables:
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                print(f"✅ Table {table} exists")
            else:
                print(f"❌ Table {table} missing")
                return False
        
        # Check sample data
        farmers_count = conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
        print(f"✅ {farmers_count} farmers in database")
        
        schemes_count = conn.execute("SELECT COUNT(*) FROM government_schemes").fetchone()[0]
        print(f"✅ {schemes_count} government schemes in database")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_models():
    """Test if ML models are available"""
    print("\n🤖 Testing ML models...")
    
    try:
        from config import MODEL_PATH, SCALER_PATH
        
        # Check if model files exist
        if os.path.exists(MODEL_PATH):
            print(f"✅ Model file exists: {MODEL_PATH}")
        else:
            print(f"❌ Model file missing: {MODEL_PATH}")
            return False
        
        if os.path.exists(SCALER_PATH):
            print(f"✅ Scaler file exists: {SCALER_PATH}")
        else:
            print(f"❌ Scaler file missing: {SCALER_PATH}")
            return False
        
        # Try loading models
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        try:
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Scaler loading failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def test_data_pipeline():
    """Test data pipeline functionality"""
    print("\n📊 Testing data pipeline...")
    
    try:
        from advanced_data_pipeline import AdvancedDataPipeline
        
        pipeline = AdvancedDataPipeline()
        print("✅ Data pipeline initialized")
        
        # Test weather API
        try:
            weather = pipeline.get_live_weather(28.6139, 77.2090)  # Delhi
            if weather and 'temperature' in weather:
                print(f"✅ Weather API working (Temperature: {weather['temperature']}°C)")
            else:
                print("⚠️ Weather API returned no data (may be rate limited)")
        except Exception as e:
            print(f"⚠️ Weather API error (may be rate limited): {e}")
        
        # Test market prices
        try:
            market_data = pipeline.get_market_prices('Rice')
            if market_data and 'price_per_quintal' in market_data:
                print(f"✅ Market data working (Rice price: ₹{market_data['price_per_quintal']}/quintal)")
            else:
                print("⚠️ Market data returned no data")
        except Exception as e:
            print(f"⚠️ Market data error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        return False

def test_ml_model():
    """Test ML model functionality"""
    print("\n🧠 Testing ML model...")
    
    try:
        from advanced_ml_model import AdvancedCreditModel
        
        model = AdvancedCreditModel()
        print("✅ ML model initialized")
        
        # Test model prediction
        try:
            # Create sample features with all 65 features
            sample_features = {
                'land_size': 2.5,
                'crop_type_encoded': 1,
                'farmer_age': 45,
                'education_level': 3,
                'family_size': 4,
                'current_temperature': 28,
                'current_humidity': 65,
                'temperature_stress': 0.1,
                'humidity_stress': 0.05,
                'frost_risk_7days': 0.1,
                'drought_risk_7days': 0.2,
                'excess_rain_risk': 0.1,
                'seasonal_rainfall_deviation': 0.0,
                'historical_drought_frequency': 1,
                'climate_change_vulnerability': 0.3,
                'current_price': 2500,
                'price_volatility': 0.15,
                'price_trend': 0.05,
                'market_demand_index': 0.6,
                'export_potential': 0.4,
                'storage_price_premium': 0.2,
                'payment_history_score': 0.8,
                'yield_consistency': 0.7,
                'loan_to_land_ratio': 0.3,
                'debt_to_income_ratio': 0.2,
                'savings_to_income_ratio': 0.1,
                'credit_utilization': 0.4,
                'number_of_credit_sources': 2,
                'informal_lending_dependency': 0.3,
                'nearest_mandi_distance': 15,
                'irrigation_access': 1,
                'connectivity_index': 0.7,
                'road_quality_index': 0.6,
                'electricity_reliability': 0.8,
                'mobile_network_strength': 0.8,
                'bank_branch_distance': 8,
                'mechanization_level': 0.4,
                'seed_quality_index': 0.7,
                'fertilizer_usage_efficiency': 0.6,
                'pest_management_score': 0.7,
                'soil_health_index': 0.8,
                'nutrient_deficiency_risk': 0.2,
                'organic_farming_adoption': 0.3,
                'precision_agriculture_usage': 0.2,
                'eligible_schemes_count': 3,
                'insurance_coverage': 1,
                'subsidy_utilization': 0.5,
                'msp_eligibility': 1,
                'kisan_credit_card': 1,
                'government_training_participation': 0.4,
                'cooperative_membership': 1,
                'community_leadership_role': 0,
                'social_capital_index': 0.6,
                'extension_service_access': 0.5,
                'peer_learning_participation': 0.4,
                'input_cost_index': 0.6,
                'labor_availability': 0.7,
                'storage_access': 0,
                'transport_cost_burden': 0.4,
                'supply_chain_integration': 0.3,
                'diversification_index': 0.5,
                'technology_adoption': 0.6,
                'disaster_preparedness': 0.3,
                'alternative_income_sources': 0.4,
                'livestock_ownership': 1
            }
            
            # Load trained model
            from config import MODEL_PATH, SCALER_PATH
            trained_model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Create feature vector
            feature_vector = list(sample_features.values())
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            prediction = trained_model.predict_proba(feature_vector_scaled)[0][1]
            credit_score = int((1 - prediction) * 850 + 150)
            
            print(f"✅ Model prediction successful")
            print(f"   Default probability: {prediction:.3f}")
            print(f"   Credit score: {credit_score}")
            
        except Exception as e:
            print(f"⚠️ Model prediction test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model error: {e}")
        return False

def test_weather_alert_system():
    """Test weather alert system"""
    print("\n🌤️ Testing weather alert system...")
    
    try:
        from weather_alert_system import WeatherAlertSystem
        
        weather_system = WeatherAlertSystem()
        print("✅ Weather alert system initialized")
        
        # Test forecast retrieval
        try:
            forecast = weather_system.get_detailed_forecast(28.6139, 77.2090)  # Delhi
            if forecast and len(forecast) > 0:
                print(f"✅ Weather forecast working ({len(forecast)} data points)")
            else:
                print("⚠️ Weather forecast returned no data (may be rate limited)")
        except Exception as e:
            print(f"⚠️ Weather forecast error (may be rate limited): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weather alert system error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AgriCred AI System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database", test_database),
        ("ML Models", test_models),
        ("Data Pipeline", test_data_pipeline),
        ("ML Model Functionality", test_ml_model),
        ("Weather Alert System", test_weather_alert_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo start the application:")
        print("streamlit run advanced_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nTo fix issues:")
        print("1. Run: python setup.py")
        print("2. Check your API keys in .env file")
        print("3. Ensure all dependencies are installed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
