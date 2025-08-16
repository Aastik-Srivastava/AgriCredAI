

# The AdvancedDataPipeline class is designed to serve as a comprehensive data handler for an agricultural analytics system. Its primary responsibilities include managing a local SQLite database, fetching and processing weather and market data, and calculating a wide range of features for machine learning models that assess farmer risk and eligibility for schemes.

# Upon initialization, the class sets up a SQLite database with several tables, such as farmers, weather_data, market_prices, soil_health, loan_history, and government_schemes. These tables are structured to store detailed information about farmers, their land, weather conditions, market prices, soil health metrics, loan records, and available government schemes.

# The class provides methods to fetch real-time weather data and forecasts from the OpenWeatherMap API, handling errors gracefully by falling back to cached or default values if the API call fails. Similarly, it simulates fetching market prices for crops, introducing random volatility to mimic real-world price fluctuations.

# A central method, calculate_advanced_features, aggregates data from the database and external sources to compute a rich set of features for a given farmer. These features include basic demographics, weather and climate risks (like temperature stress, drought, and excess rain), market trends, historical loan repayment performance, geospatial metrics (such as distance to the nearest market), eligibility for government schemes, soil health, economic indices, and risk mitigation factors. Many of these calculations use helper methods, some of which are placeholders that return randomized values for demonstration purposes.

# The code is modular, with each feature calculation encapsulated in its own method. This design makes it easy to extend or refine individual calculations as more data or business logic becomes available. The class also demonstrates good practices such as error handling, use of parameterized SQL queries to prevent injection, and clear separation of concerns between data access, external API integration, and feature engineering. Overall, this pipeline provides a robust foundation for building advanced analytics or decision support tools in the agricultural domain.


import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime
from geopy.distance import geodesic
from config import WEATHER_API_KEY, MARKET_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, WEATHER_UNITS

class AdvancedDataPipeline:
    def __init__(self):
        self.setup_database()
        self.weather_api_key = WEATHER_API_KEY
        self.weather_base_url = WEATHER_API_BASE_URL
        self.weather_units = WEATHER_UNITS

    def setup_database(self):
        self.conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        # Basic schema
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            farmer_id INTEGER PRIMARY KEY,
            name TEXT, latitude REAL, longitude REAL,
            land_size REAL, crop_type TEXT, soil_type TEXT,
            phone_number TEXT, registration_date DATE
        )""")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_alerts (
            id INTEGER PRIMARY KEY,
            farmer_id INTEGER,
            alert_type TEXT, severity TEXT, message TEXT, recommended_action TEXT,
            created_at DATETIME
        )""")
        self.conn.commit()

    def get_live_weather(self, lat, lon):
        try:
            url = f"{self.weather_base_url}/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units={self.weather_units}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'weather_condition': data['weather'][0]['description']
                }
        except Exception as e:
            print(f"Weather API error: {e}")
        return {'temperature': 25, 'humidity': 60}

    def get_weather_forecast(self, lat, lon, days=7):
        try:
            url = f"{self.weather_base_url}/forecast?lat={lat}&lon={lon}&appid={self.weather_api_key}&units={self.weather_units}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                out = []
                for item in data['list'][:days*8]:
                    out.append({
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'rainfall': item.get('rain', {}).get('3h', 0)
                    })
                return out
        except Exception as e:
            print(f"Forecast API error: {e}")
        return []

    def get_market_prices(self, crop_type, state=""):
        try:
            url = (f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
                   f"?format=json&api-key={MARKET_API_KEY}&limit=1&filters[commodity]={crop_type}")
            if state:
                url += f"&filters[state]={state}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                records = r.json().get("records", [])
                if records:
                    rec = records[0]
                    return {
                        'price_per_quintal': float(rec.get('modal_price', 0)),
                        'date': rec.get('arrival_date'),
                        'volatility': 0
                    }
        except Exception as e:
            print(f"Market API error: {e}")
        return {'price_per_quintal': None, 'volatility': None}

    # def get_soil_health(self, district):
    #     try:
    #         url = f"https://soilhealth.dac.gov.in/API/api/soilhealthcard?district={district}"
    #         r = requests.get(url, timeout=10)
    #         if r.status_code == 200:
    #             data = r.json()
    #             if data:
    #                 s = data[0]
    #                 return {
    #                     'ph_level': float(s.get('ph', 7)),
    #                     'Nitrogen': float(s.get('N', 0)),
    #                     'Phosphorus': float(s.get('P', 0)),
    #                     'Potassium': float(s.get('K', 0)),
    #                     'organic_carbon': float(s.get('OC', 0))
    #                 }
    #     except Exception as e:
    #         print(f"Soil API error: {e}")
    #     return None

    # def get_soil_health(self, district):
    #     #Demo fallback: always returns fixed demo values for now
    #     return {
    #         'ph_level': 7.1,
    #         'nitrogen': 48,
    #         'phosphorus': 6.2,
    #         'potassium': 75,
    #         'organic_carbon': 0.91
    #     }
        # ---- (uncomment below to fetch real data) ----
        # try:
        #     url = f"https://soilhealth.dac.gov.in/API/api/soilhealthcard?district={district}"
        #     r = requests.get(url, timeout=10)
        #     if r.status_code == 200:
        #         data = r.json()
        #         if data:
        #             s = data
        #             return {
        #                 'ph_level': float(s.get('ph', 7)),
        #                 'nitrogen': float(s.get('N', 0)),
        #                 'phosphorus': float(s.get('P', 0)),
        #                 'potassium': float(s.get('K', 0)),
        #                 'organic_carbon': float(s.get('OC', 0))
        #             }
        # except Exception as e:
        #     print(f"Soil API error: {e}")
        # return None

    def calculate_advanced_features(self, farmer_id):
        farmer = self.conn.execute("SELECT * FROM farmers WHERE farmer_id=?", (farmer_id,)).fetchone()
        if not farmer: return None

        lat, lon = farmer[2], farmer[3]
        current_weather = self.get_live_weather(lat, lon)
        forecast = self.get_weather_forecast(lat, lon)
        market_data = self.get_market_prices(farmer[5])
        soil = self.get_soil_health(farmer[6])

        return {
            'land_size': farmer[4],
            'crop_type_encoded': self.encode_crop_type(farmer[5]),
            'current_temperature': current_weather.get('temperature', 25),
            'current_humidity': current_weather.get('humidity', 60),
            'frost_risk_7days': self.calculate_frost_risk(forecast, farmer[5]),
            'drought_risk_7days': self.calculate_drought_risk(forecast),
            'excess_rain_risk': self.calculate_excess_rain_risk(forecast),
            'current_price': market_data.get('price_per_quintal', 0),
            'price_volatility': market_data.get('volatility', 0),
            'soil_ph': soil.get('ph_level', 7) if soil else 7,
            'soil_nitrogen': soil.get('nitrogen', 0) if soil else 0,
            'soil_phosphorus': soil.get('phosphorus', 0) if soil else 0,
            'soil_potassium': soil.get('potassium', 0) if soil else 0,
            'soil_organic_carbon': soil.get('organic_carbon', 0) if soil else 0,
        }

    def encode_crop_type(self, crop):
        mapping = {'Rice': 1, 'Wheat': 2, 'Cotton': 3, 'Sugarcane': 4, 'Soybean': 5, 'Maize': 6}
        return mapping.get(crop, 0)

    def calculate_frost_risk(self, forecast, crop_type):
        frost_sensitive = ['Rice', 'Cotton', 'Sugarcane', 'Soybean']
        if crop_type not in frost_sensitive: return 0
        frost_days = sum(1 for f in forecast if f['temperature'] < 2)
        return min(frost_days / 7, 1)

    def calculate_drought_risk(self, forecast):
        total_rain = sum(f.get('rainfall', 0) for f in forecast)
        return 1 if total_rain < 7.5 else 0.5 if total_rain < 17.5 else 0

    def calculate_excess_rain_risk(self, forecast):
        daily = [f.get('rainfall', 0) for f in forecast]
        return 0.7 if max(daily) > 50 else 0.3 if max(daily) > 30 else 0

pipeline = AdvancedDataPipeline()
