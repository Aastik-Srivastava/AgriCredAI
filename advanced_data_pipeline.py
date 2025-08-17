

# The AdvancedDataPipeline class is designed to serve as a comprehensive data handler for an agricultural analytics system. Its primary responsibilities include managing a local SQLite database, fetching and processing weather and market data, and calculating a wide range of features for machine learning models that assess farmer risk and eligibility for schemes.

# Upon initialization, the class sets up a SQLite database with several tables, such as farmers, weather_data, market_prices, soil_health, loan_history, and government_schemes. These tables are structured to store detailed information about farmers, their land, weather conditions, market prices, soil health metrics, loan records, and available government schemes.

# The class provides methods to fetch real-time weather data and forecasts from the OpenWeatherMap API, handling errors gracefully by falling back to cached or default values if the API call fails. Similarly, it simulates fetching market prices for crops, introducing random volatility to mimic real-world price fluctuations.

# A central method, calculate_advanced_features, aggregates data from the database and external sources to compute a rich set of features for a given farmer. These features include basic demographics, weather and climate risks (like temperature stress, drought, and excess rain), market trends, historical loan repayment performance, geospatial metrics (such as distance to the nearest market), eligibility for government schemes, soil health, economic indices, and risk mitigation factors. Many of these calculations use helper methods, some of which are placeholders that return randomized values for demonstration purposes.

# The code is modular, with each feature calculation encapsulated in its own method. This design makes it easy to extend or refine individual calculations as more data or business logic becomes available. The class also demonstrates good practices such as error handling, use of parameterized SQL queries to prevent injection, and clear separation of concerns between data access, external API integration, and feature engineering. Overall, this pipeline provides a robust foundation for building advanced analytics or decision support tools in the agricultural domain.


import pandas as pd
import numpy as np
import requests
import sqlite3
import json
import hashlib
import logging
from datetime import datetime, timedelta
from config import (WEATHER_API_KEY, MARKET_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, 
                   WEATHER_UNITS, CACHE_ENABLED, CACHE_TTL, RATE_LIMIT_ENABLED, 
                   RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD, MARKET_API_BASE_URL)

# Setup logging
logger = logging.getLogger("market_data")
logger.setLevel(logging.INFO)

# Remove any existing handlers (to avoid duplicate logs)
if logger.hasHandlers():
    logger.handlers.clear()

# StreamHandler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(stream_handler)

# FileHandler (line-by-line log file)
file_handler = logging.FileHandler("market_data.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

class AdvancedDataPipeline:
    def __init__(self):
        self.setup_database()
        self.setup_market_cache()  # Add this line
        self.weather_api_key = WEATHER_API_KEY
        self.weather_base_url = WEATHER_API_BASE_URL
        self.weather_units = WEATHER_UNITS
        self.market_api_key = MARKET_API_KEY
        self.rate_limit_count = 0
        self.rate_limit_reset = datetime.now() + timedelta(seconds=RATE_LIMIT_PERIOD)

    def setup_database(self):
        self.conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        
        # Enhanced farmers table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            farmer_id INTEGER PRIMARY KEY,
            name TEXT, latitude REAL, longitude REAL,
            land_size REAL, crop_type TEXT, soil_type TEXT,
            phone_number TEXT, registration_date DATE,
            education_level INTEGER, family_size INTEGER,
            irrigation_access INTEGER, cooperative_membership INTEGER,
            insurance_coverage INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Loans table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS loans (
            loan_id INTEGER PRIMARY KEY,
            farmer_id INTEGER,
            amount REAL, interest_rate REAL, duration_months INTEGER,
            disbursed_date DATE, due_date DATE, 
            status TEXT, -- 'active', 'repaid', 'defaulted'
            repaid_amount REAL DEFAULT 0,
            credit_score INTEGER,
            risk_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers (farmer_id)
        )""")
        
        # Portfolio metrics time series table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_metrics (
            id INTEGER PRIMARY KEY,
            date DATE UNIQUE,
            total_farmers INTEGER,
            total_loans INTEGER,
            total_portfolio_value REAL,
            active_loans INTEGER,
            repaid_loans INTEGER,
            defaulted_loans INTEGER,
            default_rate REAL,
            avg_credit_score REAL,
            total_land_size REAL,
            avg_loan_amount REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Weather alerts (your existing table)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_alerts (
            id INTEGER PRIMARY KEY,
            farmer_id INTEGER,
            alert_type TEXT, severity TEXT, message TEXT, recommended_action TEXT,
            created_at DATETIME
        )""")
        
        self.conn.commit()

    def seed_portfolio_history(self, days=60):
        """Seeds portfolio_metrics with very smooth, believable time series."""
        import numpy as np
        from datetime import date, timedelta

        today = date.today()

        # Starting values close to reality
        start_portfolio = 2_000_000
        daily_growth = 34000  # Lower daily growth for smoother effect

        start_defaults = 11.7
        final_defaults = 5.2

        start_score = 565
        final_score = 710

        np.random.seed(42)  # For reproducible demo trends

        portfolio_value = start_portfolio
        default_rate = start_defaults
        avg_credit_score = start_score

        for i in range(days):
            cur_date = today - timedelta(days=days - i)

            # Smoothly trend portfolio value & add tiny noise
            portfolio_value += daily_growth + np.random.normal(scale=3100)
            portfolio_value = max(portfolio_value, 0)

            # Smoothly decrease default rate
            default_rate = start_defaults - ((start_defaults - final_defaults) / days) * i
            default_rate += np.random.normal(scale=0.07)  # very light jitter
            default_rate = max(3.5, min(default_rate, 20))

            # Smooth rise in avg credit score
            avg_credit_score = start_score + ((final_score - start_score) / days) * i
            avg_credit_score += np.random.normal(scale=2.7)
            avg_credit_score = max(500, min(avg_credit_score, 850))

            total_loans = 415 + i * 3 + int(np.random.normal(scale=1.5))
            total_loans = int(max(300, total_loans))

            total_farmers = 305 + int(i * 2.4 + np.random.normal(scale=1.8))
            active_loans = int(total_loans * 0.735 + np.random.normal(scale=2))
            repaid_loans = int(total_loans * 0.24 + np.random.normal(scale=1))
            defaulted_loans = int(total_loans * default_rate / 100)
            avg_loan_amount = portfolio_value / (total_loans if total_loans > 0 else 1)
            total_land_size = 2400 + (i * 2.95) + np.random.normal(scale=1)

            # Insert metrics
            self.conn.execute("""
                INSERT OR REPLACE INTO portfolio_metrics (
                    date, total_farmers, total_loans, total_portfolio_value, active_loans, 
                    repaid_loans, defaulted_loans, default_rate, avg_credit_score, 
                    total_land_size, avg_loan_amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(cur_date), int(total_farmers), int(total_loans), float(portfolio_value),
                int(active_loans), int(repaid_loans), int(defaulted_loans),
                float(default_rate), float(avg_credit_score), float(total_land_size), float(avg_loan_amount)
            ))

        self.conn.commit()
        print(f"Seeded {days} days with SMOOTH demo portfolio trends.")

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
        """Get market prices with corrected data.gov.in API format"""
        
        # Standardize crop name for API (data.gov.in expects proper case)
        crop_mapping = {
            'wheat': 'Wheat', 
            'rice': 'Rice', 
            'cotton': 'Cotton',
            'soybean': 'Soybean', 
            'maize': 'Maize', 
            'sugarcane': 'Sugarcane',
            'potato': 'Potato',
            'onion': 'Onion',
            'tomato': 'Tomato'
        }
        
        standardized_crop = crop_mapping.get(crop_type.lower(), crop_type.title())
        
        # Create cache key
        crop_hash = hashlib.md5(f"{standardized_crop.lower()}:{state.lower()}".encode()).hexdigest()
        
        # Check cache first if enabled
        if CACHE_ENABLED:
            cached_data = self._get_cached_market_data(crop_hash)
            if cached_data:
                logger.info(f"Using cached market data for {standardized_crop} in {state or 'all states'}")
                return cached_data
        
        # Check rate limit
        if RATE_LIMIT_ENABLED and not self._check_market_rate_limit():
            logger.warning("Market API rate limit exceeded, using fallback data")
            return self._get_fallback_market_data(crop_type, state)
        
        try:
            # Use correct data.gov.in resource ID for market data
            resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
            base_url = f"https://api.data.gov.in/resource/{resource_id}"
            
            # Build parameters properly
            params = {
                'api-key': self.market_api_key,
                'format': 'json',
                'limit': 5  # Get multiple records for better data
            }
            
            # Add filters with correct format
            if standardized_crop:
                params[f'filters[commodity]'] = standardized_crop
            if state and state.lower() != "all":
                params[f'filters[state]'] = state.title()
                
            logger.info(f"Fetching market data for {standardized_crop} in {state or 'all states'}")
            logger.debug(f"API URL: {base_url}")
            logger.debug(f"Parameters: {params}")
            
            # Make the API call with proper headers
            headers = {
                'User-Agent': 'Agricultural-Credit-Intelligence/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            
            # Increment rate limit counter
            if RATE_LIMIT_ENABLED:
                self.rate_limit_count += 1
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.debug(f"Raw API response: {json.dumps(data, indent=2)}")
                    
                    # Handle different response structures
                    records = []
                    if isinstance(data, dict):
                        records = data.get("records", [])
                        if not records and "data" in data:
                            records = data["data"]
                    elif isinstance(data, list):
                        records = data
                    
                    if records:
                        # Use the most recent record
                        record = records[0] if isinstance(records[0], dict) else records[0]
                        
                        # Extract price data with proper field mapping
                        market_data = {
                            'price_per_quintal': float(record.get('modal_price', record.get('modalPrice', 0))),
                            'min_price': float(record.get('min_price', record.get('minPrice', 0))), 
                            'max_price': float(record.get('max_price', record.get('maxPrice', 0))),
                            'market': record.get('market', record.get('marketName', '')),
                            'district': record.get('district', ''),
                            'state': record.get('state', ''),
                            'date': record.get('arrival_date', record.get('arrivalDate', record.get('date', ''))),
                            'variety': record.get('variety', ''),
                            'commodity': record.get('commodity', ''),
                            'volatility': 0,  # Calculate if historical data available
                            'source': 'data.gov.in API'
                        }
                        
                        # Cache successful result
                        if CACHE_ENABLED:
                            self._cache_market_data(crop_hash, standardized_crop, state, market_data)
                        
                        logger.info(f"Successfully fetched market data: {standardized_crop} @ ₹{market_data['price_per_quintal']}/quintal")
                        return market_data
                    else:
                        logger.warning(f"No records found in API response for {standardized_crop}")
                        logger.debug(f"Full response: {response.text}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {response.text}")
                    
            elif response.status_code == 403:
                logger.error("API Access Denied - Check your API key permissions")
                logger.debug(f"Response: {response.text}")
                
            elif response.status_code == 404:
                logger.error("API endpoint not found - Check resource ID")
                logger.debug(f"URL: {base_url}")
                
            else:
                logger.error(f"API returned status {response.status_code}")
                logger.debug(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to data.gov.in API")
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback to simulated data
        logger.info(f"Using fallback data for {standardized_crop}")
        return self._get_fallback_market_data(crop_type, state)

    def get_soil_health(self, district):
        #Demo fallback: always returns fixed demo values for now
        return {
            'ph_level': 7.1,
            'nitrogen': 48,
            'phosphorus': 6.2,
            'potassium': 75,
            'organic_carbon': 0.91
        }
       #get real data from some source


    def calculate_and_store_portfolio_metrics(self):
        """Calculate current portfolio metrics and store in time series table"""
        from datetime import date
        
        today = date.today()
        
        # Calculate metrics
        total_farmers = self.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
        
        loan_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_loans,
                SUM(amount) as total_portfolio,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_loans,
                SUM(CASE WHEN status = 'repaid' THEN 1 ELSE 0 END) as repaid_loans,
                SUM(CASE WHEN status = 'defaulted' THEN 1 ELSE 0 END) as defaulted_loans,
                AVG(credit_score) as avg_credit_score,
                AVG(amount) as avg_loan_amount
            FROM loans
        """).fetchone()
        
        total_land = self.conn.execute("SELECT SUM(land_size) FROM farmers").fetchone()[0]
        
        total_loans, total_portfolio, active_loans, repaid_loans, defaulted_loans, avg_credit_score, avg_loan_amount = loan_stats
        
        default_rate = (defaulted_loans / max(total_loans, 1)) * 100
        
        # Store or update today's metrics
        self.conn.execute("""
            INSERT OR REPLACE INTO portfolio_metrics 
            (date, total_farmers, total_loans, total_portfolio_value, active_loans, 
            repaid_loans, defaulted_loans, default_rate, avg_credit_score, 
            total_land_size, avg_loan_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (today, total_farmers, total_loans, total_portfolio or 0, active_loans,
            repaid_loans, defaulted_loans, default_rate, avg_credit_score or 0,
            total_land or 0, avg_loan_amount or 0))
        
        self.conn.commit()
        return {
            'total_farmers': total_farmers,
            'total_loans': total_loans,
            'total_portfolio': total_portfolio or 0,
            'active_loans': active_loans,
            'default_rate': default_rate,
            'avg_credit_score': avg_credit_score or 0
        }

    def get_portfolio_trends(self, days=30):
        """Get portfolio metrics trends over time"""
        from datetime import date, timedelta
        
        start_date = date.today() - timedelta(days=days)
        
        trends = self.conn.execute("""
            SELECT date, total_farmers, total_loans, total_portfolio_value, 
                default_rate, avg_credit_score
            FROM portfolio_metrics 
            WHERE date >= ?
            ORDER BY date
        """, (start_date,)).fetchall()
        
        return pd.DataFrame(trends, columns=[
            'date', 'total_farmers', 'total_loans', 'total_portfolio_value',
            'default_rate', 'avg_credit_score'
        ])



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


    def seed_farmers(self, num_farmers=100):
        """Create realistic farmer data and store in database"""
        import random
        from datetime import datetime, timedelta
        
        # Check if farmers already exist
        existing = self.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
        if existing >= num_farmers:
            print(f"Database already has {existing} farmers")
            return
        
        # Sample data for realistic farmers
        names = ["Raj Kumar", "Suresh Patel", "Ramesh Singh", "Vijay Sharma", "Anil Verma", 
                 "Prakash Rao", "Dinesh Kumar", "Mahesh Gupta", "Santosh Yadav", "Ravi Joshi"]
        crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Soybean", "Maize"]
        soil_types = ["Loamy", "Clay", "Sandy", "Black", "Red", "Alluvial"]
        
        # Indian coordinates (rough)
        lat_range = (8.4, 37.6)  # India's latitude range
        lon_range = (68.7, 97.4)  # India's longitude range
        
        farmers_data = []
        for i in range(num_farmers - existing):
            farmer_data = (
                f"{random.choice(names)} {i+existing+1}",
                round(random.uniform(lat_range[0], lat_range[1]), 4),   # FIXED: lat_range, lat_range[1]
                round(random.uniform(lon_range[0], lon_range[1]), 4),   # FIXED: lon_range, lon_range[1]
                round(np.random.gamma(2, 1.5), 2),  # land_size
                random.choice(crops),
                random.choice(soil_types),
                f"98{random.randint(10000000, 99999999)}",  # phone
                (datetime.now() - timedelta(days=random.randint(1, 365))).date(),  # registration
                random.randint(1, 5),  # education_level
                random.randint(2, 8),  # family_size
                random.choice([0, 1]),  # irrigation_access
                random.choice([0, 1]),  # cooperative_membership  
                random.choice([0, 1])   # insurance_coverage
            )
            farmers_data.append(farmer_data)
        
        # Insert all farmers
        self.conn.executemany("""
            INSERT INTO farmers (name, latitude, longitude, land_size, crop_type, soil_type, 
                               phone_number, registration_date, education_level, family_size, 
                               irrigation_access, cooperative_membership, insurance_coverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, farmers_data)
        self.conn.commit()
        print(f"Seeded {len(farmers_data)} new farmers. Total: {existing + len(farmers_data)}")

    def seed_loans_for_farmers(self):
        """Create loan records for existing farmers"""
        from datetime import datetime, timedelta
        import random
        
        # Get all farmers
        farmers = self.conn.execute("SELECT farmer_id FROM farmers").fetchall()
        
        loans_data = []
        for farmer_id, in farmers:
            # 70% chance a farmer has taken a loan
            if random.random() < 0.7:
                amount = round(random.uniform(50000, 500000), 2)
                interest_rate = round(random.uniform(8.5, 14.5), 2)
                duration = random.choice([12, 24, 36, 48, 60])
                disbursed_date = datetime.now() - timedelta(days=random.randint(30, 730))
                due_date = disbursed_date + timedelta(days=duration * 30)
                status = random.choices(['active', 'repaid', 'defaulted'], weights=[60, 35, 5])[0]
                
                repaid_amount = 0
                if status == 'repaid':
                    repaid_amount = amount * (1 + interest_rate/100 * duration/12)
                elif status == 'defaulted':
                    repaid_amount = amount * random.uniform(0.2, 0.8)
                
                credit_score = random.randint(300, 850)
                risk_level = 'Low' if credit_score > 700 else 'Medium' if credit_score > 500 else 'High'
                
                loans_data.append((
                    farmer_id, amount, interest_rate, duration,
                    disbursed_date.date(), due_date.date(), status,
                    repaid_amount, credit_score, risk_level
                ))
        
        self.conn.executemany("""
            INSERT INTO loans (farmer_id, amount, interest_rate, duration_months,
                             disbursed_date, due_date, status, repaid_amount, credit_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, loans_data)
        self.conn.commit()
        print(f"Seeded {len(loans_data)} loans")

    def calculate_and_store_portfolio_metrics(self):
        """Calculate current portfolio metrics and store in time series table"""
        from datetime import date
        
        today = date.today()
        
        # Calculate metrics with safe defaults
        total_farmers = self.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0] or 0
        total_land = self.conn.execute("SELECT COALESCE(SUM(land_size), 0) FROM farmers").fetchone()[0] or 0
        
        # Check if loans table has any data
        loan_count = self.conn.execute("SELECT COUNT(*) FROM loans").fetchone()[0] or 0
        
        if loan_count == 0:
            # No loans yet - set all loan metrics to 0
            total_loans = 0
            total_portfolio = 0
            active_loans = 0
            repaid_loans = 0
            defaulted_loans = 0
            avg_credit_score = 0
            avg_loan_amount = 0
            default_rate = 0
        else:
            # Get loan statistics
            loan_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_loans,
                    COALESCE(SUM(amount), 0) as total_portfolio,
                    COALESCE(SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END), 0) as active_loans,
                    COALESCE(SUM(CASE WHEN status = 'repaid' THEN 1 ELSE 0 END), 0) as repaid_loans,
                    COALESCE(SUM(CASE WHEN status = 'defaulted' THEN 1 ELSE 0 END), 0) as defaulted_loans,
                    COALESCE(AVG(credit_score), 0) as avg_credit_score,
                    COALESCE(AVG(amount), 0) as avg_loan_amount
                FROM loans
            """).fetchone()
            
            total_loans = loan_stats[0] or 0
            total_portfolio = loan_stats[1] or 0
            active_loans = loan_stats[2] or 0
            repaid_loans = loan_stats[3] or 0
            defaulted_loans = loan_stats[4] or 0
            avg_credit_score = loan_stats[5] or 0
            avg_loan_amount = loan_stats[6] or 0
            
            # Calculate default rate safely
            default_rate = (defaulted_loans / total_loans * 100) if total_loans > 0 else 0
        
            self.conn.execute("""
            INSERT OR REPLACE INTO portfolio_metrics 
            (date, total_farmers, total_loans, total_portfolio_value, active_loans, 
            repaid_loans, defaulted_loans, default_rate, avg_credit_score, 
            total_land_size, avg_loan_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(today),
            int(total_farmers),
            int(total_loans),
            float(total_portfolio),
            int(active_loans),
            int(repaid_loans),
            int(defaulted_loans),
            float(default_rate),
            float(avg_credit_score),
            float(total_land),
            float(avg_loan_amount)
        ))

        
        self.conn.commit()
        
        return {
            'total_farmers': total_farmers,
            'total_loans': total_loans,
            'total_portfolio': total_portfolio,
            'active_loans': active_loans,
            'default_rate': default_rate,
            'avg_credit_score': avg_credit_score
        }

    def get_portfolio_trends(self, days=30):
        """Get portfolio metrics trends over time"""
        from datetime import date, timedelta
        
        start_date = date.today() - timedelta(days=days)
        
        trends = self.conn.execute("""
            SELECT date, total_farmers, total_loans, total_portfolio_value, 
                   default_rate, avg_credit_score
            FROM portfolio_metrics 
            WHERE date >= ?
            ORDER BY date
        """, (start_date,)).fetchall()
        
        return pd.DataFrame(trends, columns=[
            'date', 'total_farmers', 'total_loans', 'total_portfolio_value',
            'default_rate', 'avg_credit_score'
        ])



    def _get_cached_market_data(self, crop_hash):
        """Get market data from cache if available and not expired"""
        try:
            cur = self.conn.execute(
                "SELECT market_data, source FROM market_data_cache WHERE crop_hash = ? AND expires_at > ?", 
                (crop_hash, datetime.now())
            )
            result = cur.fetchone()
            if result:
                market_data = json.loads(result[0])
                market_data['source'] = result[1]  # Add data provenance
                return market_data
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached market data: {e}")
            return None

    def _cache_market_data(self, crop_hash, crop_type, state, market_data):
        """Cache market data for future use"""
        try:
            market_json = json.dumps(market_data)
            expires_at = datetime.now() + timedelta(seconds=CACHE_TTL)
            
            self.conn.execute(
                "INSERT OR REPLACE INTO market_data_cache "
                "(crop_hash, crop_type, state, market_data, source, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (crop_hash, crop_type, state, market_json, market_data['source'], 
                 datetime.now(), expires_at)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error caching market data: {e}")

    def _check_market_rate_limit(self):
        """Check if rate limit is exceeded. Returns True if within limit, False if exceeded."""
        now = datetime.now()
        if now > self.rate_limit_reset:
            # Reset counter if period has passed
            self.rate_limit_count = 0
            self.rate_limit_reset = now + timedelta(seconds=RATE_LIMIT_PERIOD)
            return True
        
        return self.rate_limit_count < RATE_LIMIT_CALLS

    def _get_fallback_market_data(self, crop_type, state):
        """Generate realistic fallback market data based on crop type"""
        import random
        
        # Base prices for common crops (in INR per quintal)
        base_prices = {
            'Rice': 2000,
            'Wheat': 1800,
            'Cotton': 6000,
            'Sugarcane': 300,
            'Soybean': 3800,
            'Maize': 1600,
            'Potato': 1200,
            'Onion': 1500,
            'Tomato': 1800
        }
        
        # Use the base price for the crop or a default value
        base_price = base_prices.get(crop_type, 2000)
        
        # Add some randomness to make it realistic
        modal_price = base_price * (1 + random.uniform(-0.1, 0.1))
        min_price = modal_price * (1 - random.uniform(0.05, 0.15))
        max_price = modal_price * (1 + random.uniform(0.05, 0.15))
        
        # Generate a realistic date (within last 7 days)
        days_ago = random.randint(0, 7)
        date_str = (datetime.now() - timedelta(days=days_ago)).strftime('%d/%m/%Y')
        
        # List of markets by state
        markets_by_state = {
            'Maharashtra': ['Pune', 'Nagpur', 'Mumbai', 'Nashik'],
            'Punjab': ['Amritsar', 'Ludhiana', 'Jalandhar', 'Patiala'],
            'Karnataka': ['Bangalore', 'Mysore', 'Hubli', 'Belgaum'],
            'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Varanasi', 'Agra'],
            'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem']
        }
        
        # Select a market based on state or random
        if state and state in markets_by_state:
            market = random.choice(markets_by_state[state])
            district = market  # Simplification
        else:
            all_markets = [m for markets in markets_by_state.values() for m in markets]
            market = random.choice(all_markets)
            district = market  # Simplification
        
        return {
            'price_per_quintal': round(modal_price, 2),
            'min_price': round(min_price, 2),
            'max_price': round(max_price, 2),
            'market': market,
            'district': district,
            'date': date_str,
            'variety': 'Common',
            'volatility': round(random.uniform(0.02, 0.15), 2),
            'source': 'fallback_simulation'
        }

    def setup_market_cache(self):
        """Create market_data_cache table for storing market price data"""
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS market_data_cache (
            id INTEGER PRIMARY KEY,
            crop_hash TEXT UNIQUE,
            crop_type TEXT,
            state TEXT,
            market_data TEXT,
            source TEXT,
            created_at DATETIME,
            expires_at DATETIME
        )
        ''')
        self.conn.commit()
