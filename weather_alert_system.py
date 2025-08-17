import logging
import sqlite3
from datetime import datetime, timedelta
import time
import threading
import requests
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from config import WEATHER_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, WEATHER_UNITS, CACHE_ENABLED, CACHE_TTL, RATE_LIMIT_ENABLED, RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD

# ---------- logging ----------
logger = logging.getLogger("weather_alerts")
logger.setLevel(logging.INFO)

# Remove any existing handlers (to avoid duplicate logs)
if logger.hasHandlers():
    logger.handlers.clear()

# StreamHandler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(stream_handler)

# FileHandler (line-by-line log file)
file_handler = logging.FileHandler("weather_alerts.log", mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)


# ---------- DB bootstrap ----------
def setup_alerts_table():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS weather_alerts (
        id INTEGER PRIMARY KEY,
        farmer_id INTEGER,
        alert_type TEXT,
        severity TEXT,
        message TEXT,
        recommended_action TEXT,
        created_at DATETIME,
        acknowledged BOOLEAN DEFAULT 0
    )
    ''')

    # Add "acknowledged" column if missing (try-except to avoid error if exists)
    try:
        conn.execute("ALTER TABLE weather_alerts ADD COLUMN acknowledged BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
            
    # Create weather forecast cache table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS weather_forecast_cache (
        id INTEGER PRIMARY KEY,
        location_hash TEXT UNIQUE,
        lat REAL,
        lon REAL,
        forecast_data TEXT,
        source TEXT,
        created_at DATETIME,
        expires_at DATETIME
    )
    ''')

    conn.commit()
    conn.close()


class WeatherAlertSystem:
    """
    Streamlit-friendly weather alert system:
    - No infinite loop by default.
    - Exposes run_once() and threaded monitor you can start/stop.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or WEATHER_API_KEY
        # sqlite: allow multiple threads
        self.conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.rate_limit_count = 0
        self.rate_limit_reset = datetime.now() + timedelta(seconds=RATE_LIMIT_PERIOD)





    def run_once_mvp(self) -> Dict[str, Any]:
        """MVP version: Check alerts with minimal logging and return summary"""
        try:
            farmers = self.get_all_farmers()
            if not farmers:
                return {"status": "no_farmers", "alerts": [], "summary": "No farmers found"}
            
            # Sample only first 5 farmers for MVP to avoid excessive processing
            sample_farmers = farmers[:5]
            alerts_generated = []
            farmers_checked = 0
            total_alerts = 0
            
            for farmer in sample_farmers:
                farmer_id = farmer[0]
                name = farmer[1]
                lat, lon = farmer[2], farmer[3]
                crop_type = farmer[5]
                
                if lat is None or lon is None:
                    continue
                    
                farmers_checked += 1
                
                # Get forecast with minimal logging
                forecast, source = self.get_lightweight_forecast(lat, lon)
                if not forecast:
                    continue
                
                # Generate alerts
                alerts = self.build_alerts_minimal(forecast, crop_type)
                
                if alerts:
                    # Only log when alerts are found
                    logger.info(f"Weather alerts generated for {name}: {len(alerts)} alerts")
                    
                    for alert in alerts:
                        alert['farmer_name'] = name
                        alert['farmer_id'] = farmer_id
                        alerts_generated.append(alert)
                    
                    total_alerts += len(alerts)
            
            # Single summary log entry
            logger.info(f"Weather check complete: {farmers_checked} farmers checked, {total_alerts} alerts generated")
            
            return {
                "status": "success",
                "alerts": alerts_generated,
                "summary": f"Checked {farmers_checked} farmers, found {total_alerts} weather alerts"
            }
            
        except Exception as e:
            logger.error(f"Weather alert check failed: {str(e)}")
            return {"status": "error", "alerts": [], "summary": f"Error: {str(e)}"}

    def get_lightweight_forecast(self, lat: float, lon: float) -> Tuple[List[Dict], str]:
        """Lightweight forecast with minimal API calls and logging"""
        location_hash = hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest()
        
        # Check cache first
        if CACHE_ENABLED:
            cached = self._get_cached_forecast(location_hash)
            if cached:
                return cached
        
        # For MVP, use fallback data to avoid API rate limits and excessive logging
        return self._get_fallback_forecast_minimal(lat, lon), "simulated"

    def _get_fallback_forecast_minimal(self, lat: float, lon: float) -> List[Dict]:
        """Generate minimal realistic forecast data (24 hours only)"""
        import random
        now = datetime.now()
        forecast = []
        
        # Only generate 8 entries (24 hours) instead of 56 (7 days)
        base_temp = 25 + random.uniform(-5, 5)
        
        for i in range(8):  # 24 hours of 3-hourly data
            forecast_time = now + timedelta(hours=i*3)
            temperature = base_temp + random.uniform(-3, 3)
            humidity = random.uniform(60, 85)
            rainfall = random.uniform(0, 5) if random.random() < 0.2 else 0
            
            forecast.append({
                'datetime': forecast_time,
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall,
                'wind_speed': random.uniform(5, 15)
            })
        
        return forecast

    def build_alerts_minimal(self, forecast: List[Dict], crop_type: str) -> List[Dict]:
        """Generate alerts with minimal logging"""
        alerts = []
        
        # Check only critical risks for MVP
        frost = self.check_frost_risk_minimal(forecast, crop_type)
        if frost['risk_level'] >= 0.5:  # Only high-risk alerts
            alerts.append({**frost, 'type': 'frost_warning'})
        
        drought = self.check_drought_risk_minimal(forecast)
        if drought['risk_level'] >= 0.5:
            alerts.append({**drought, 'type': 'drought_warning'})
        
        return alerts

    def check_frost_risk_minimal(self, forecast: List[Dict], crop_type: str) -> Dict:
        """Minimal frost risk check"""
        frost_sensitive = {'Rice', 'Cotton', 'Sugarcane', 'Soybean', 'Maize'}
        
        if crop_type not in frost_sensitive:
            return {'risk_level': 0, 'severity': 'low', 'message': '', 'recommended_action': ''}
        
        cold_periods = sum(1 for f in forecast if f.get('temperature', 20) < 5)
        risk_level = min(cold_periods / 3, 1.0)
        
        if risk_level > 0:
            return {
                'risk_level': risk_level,
                'severity': 'high' if risk_level > 0.7 else 'medium',
                'message': f'Frost warning: {cold_periods} periods below 5°C expected',
                'recommended_action': 'Apply frost protection measures'
            }
        
        return {'risk_level': 0, 'severity': 'low', 'message': '', 'recommended_action': ''}

    def check_drought_risk_minimal(self, forecast: List[Dict]) -> Dict:
        """Minimal drought risk check"""
        total_rain = sum(f.get('rainfall', 0) for f in forecast)
        
        if total_rain < 10:  # Very low rainfall in 24h
            return {
                'risk_level': 0.8,
                'severity': 'high',
                'message': f'Drought risk: Only {total_rain:.1f}mm rain expected in 24h',
                'recommended_action': 'Plan irrigation immediately'
            }
        elif total_rain < 20:
            return {
                'risk_level': 0.6,
                'severity': 'medium', 
                'message': f'Low rainfall: {total_rain:.1f}mm expected in 24h',
                'recommended_action': 'Monitor soil moisture levels'
            }
        
        return {'risk_level': 0.2, 'severity': 'low', 'message': '', 'recommended_action': ''}

    # ---------- Public API ----------
    def run_once(self) -> int:
        """Check all farmers once, create alerts when risks exceed thresholds. Returns number of farmers alerted."""
        try:
            farmers = self.get_all_farmers()
            if not farmers:
                logger.info("No farmers found.")
                return 0

            total_alerted = 0
            for farmer in farmers:
                # farmers schema: (0)farmer_id, (1)name, (2)lat, (3)lon, (4)land_size,
                #                 (5)crop_type, (6)soil_type, (7)phone_number, (8)registration_date
                farmer_id = farmer[0]
                name = farmer[1]
                lat, lon = farmer[2], farmer[3]
                crop_type = farmer[5]
                phone = farmer[7]

                if lat is None or lon is None:
                    logger.warning(f"Farmer {farmer_id} '{name}' has no lat/lon; skipping.")
                    continue

                forecast, source = self.get_detailed_forecast(lat, lon)
                if not forecast:
                    logger.warning(f"No forecast for farmer {farmer_id} '{name}'; skipping.")
                    continue

                alerts = self.build_alerts(forecast, crop_type)
                if alerts:
                    # Add data provenance to alerts
                    for alert in alerts:
                        alert['data_source'] = source
                    self.send_farmer_alerts(farmer_id, name, phone, alerts)
                    total_alerted += 1

            logger.info(f"run_once completed. Farmers alerted: {total_alerted}")
            return total_alerted

        except Exception as e:
            logger.exception(f"run_once error: {e}")
            return 0

    def start_background_monitor(self, interval_seconds: int = 7200):
        """Start a background thread that runs checks every interval_seconds."""
        if self.thread and self.thread.is_alive():
            logger.info("Monitor already running.")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._monitor_loop, args=(interval_seconds,), daemon=True
        )
        self.thread.start()
        logger.info("Background monitor started.")

    def stop_background_monitor(self):
        """Stop the background thread."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5)
            logger.info("Background monitor stopped.")

    # ---------- Internal loop ----------
    def _monitor_loop(self, interval_seconds: int):
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"Background monitor error: {e}")
            # Sleep in small chunks so stop_event is responsive
            slept = 0
            while slept < interval_seconds and not self.stop_event.is_set():
                time.sleep(min(5, interval_seconds - slept))
                slept += min(5, interval_seconds - slept)

    # ---------- Forecast + risks ----------
    def get_detailed_forecast(self, lat: float, lon: float) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch 3-hourly forecast (up to ~5 days) with caching, fallbacks, and data provenance.
        Returns a tuple of (forecast_data, data_source).
        """
        # Generate a unique hash for this location
        location_hash = hashlib.md5(f"{lat:.4f},{lon:.4f}".encode()).hexdigest()
        
        # Check cache if enabled
        if CACHE_ENABLED:
            cached_forecast = self._get_cached_forecast(location_hash)
            if cached_forecast:
                logger.info(f"Using cached forecast for {lat:.4f},{lon:.4f}")
                return cached_forecast
        
        # Check rate limiting
        if RATE_LIMIT_ENABLED and self._check_rate_limit():
            logger.warning("Rate limit exceeded, using fallback data")
            return self._get_fallback_forecast(lat, lon), "fallback_simulation"
        
        # Try to get real data from API
        try:
            url = f"{WEATHER_API_BASE_URL}/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units={WEATHER_UNITS}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            
            # Increment rate limit counter
            if RATE_LIMIT_ENABLED:
                self.rate_limit_count += 1
            
            lst = data.get("list", [])
            if not lst:
                logger.warning(f"Empty forecast list from API for {lat:.4f},{lon:.4f}")
                return self._get_fallback_forecast(lat, lon), "fallback_simulation"

            forecast = []
            for item in lst[:56]:  # aim for 7 days * 8 (3-hourly) if available
                main = item.get("main", {})
                wind = item.get("wind", {})
                weather = item.get("weather", [{}])[0]
                clouds = item.get("clouds", {})

                forecast.append({
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'temperature': main.get('temp'),
                    'feels_like': main.get('feels_like'),
                    'temp_min': main.get('temp_min'),
                    'temp_max': main.get('temp_max'),
                    'humidity': main.get('humidity'),
                    'pressure': main.get('pressure'),
                    'rainfall': item.get('rain', {}).get('3h', 0) or 0,
                    'wind_speed': wind.get('speed', 0),
                    'wind_direction': wind.get('deg', 0),
                    'cloud_cover': clouds.get('all', 0),
                    'weather_main': weather.get('main', ''),
                    'weather_description': weather.get('description', '')
                })
            
            # Cache the forecast if enabled
            if CACHE_ENABLED:
                self._cache_forecast(location_hash, lat, lon, forecast, "openweathermap_api")
                
            return forecast, "openweathermap_api"

        except Exception as e:
            logger.exception(f"Forecast API error: {e}")
            return self._get_fallback_forecast(lat, lon), "fallback_simulation"
    
    def _get_cached_forecast(self, location_hash: str) -> Optional[Tuple[List[Dict[str, Any]], str]]:
        """Get forecast from cache if available and not expired."""
        try:
            cur = self.conn.execute(
                "SELECT forecast_data, source FROM weather_forecast_cache WHERE location_hash = ? AND expires_at > ?", 
                (location_hash, datetime.now())
            )
            result = cur.fetchone()
            if result:
                forecast_data = json.loads(result[0])
                source = result[1]
                
                # Convert datetime strings back to datetime objects
                for item in forecast_data:
                    if 'datetime' in item and isinstance(item['datetime'], str):
                        item['datetime'] = datetime.fromisoformat(item['datetime'])
                        
                return forecast_data, source
            return None
        except Exception as e:
            logger.exception(f"Error retrieving cached forecast: {e}")
            return None
    
    def _cache_forecast(self, location_hash: str, lat: float, lon: float, 
                        forecast: List[Dict[str, Any]], source: str) -> None:
        """Cache forecast data for future use."""
        try:
            # Convert datetime objects to ISO format strings for JSON serialization
            serializable_forecast = []
            for item in forecast:
                item_copy = item.copy()
                if 'datetime' in item_copy and isinstance(item_copy['datetime'], datetime):
                    item_copy['datetime'] = item_copy['datetime'].isoformat()
                serializable_forecast.append(item_copy)
                
            forecast_json = json.dumps(serializable_forecast)
            expires_at = datetime.now() + timedelta(seconds=CACHE_TTL)
            
            self.conn.execute(
                "INSERT OR REPLACE INTO weather_forecast_cache "
                "(location_hash, lat, lon, forecast_data, source, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (location_hash, lat, lon, forecast_json, source, datetime.now(), expires_at)
            )
            self.conn.commit()
        except Exception as e:
            logger.exception(f"Error caching forecast: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded. Returns True if limit exceeded."""
        now = datetime.now()
        if now > self.rate_limit_reset:
            # Reset counter if period has passed
            self.rate_limit_count = 0
            self.rate_limit_reset = now + timedelta(seconds=RATE_LIMIT_PERIOD)
            return False
        
        return self.rate_limit_count >= RATE_LIMIT_CALLS
    
    def _get_fallback_forecast(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Generate realistic fallback weather data based on location and season."""
        import random
        now = datetime.now()
        forecast = []
        
        # Determine season-appropriate temperature range (simple approximation)
        month = now.month
        is_northern = lat > 0
        
        # Simple seasonal temperature ranges (adjust based on hemisphere)
        if (is_northern and (month >= 3 and month <= 5)) or (not is_northern and (month >= 9 and month <= 11)):
            # Spring/Autumn
            base_temp = 15 + random.uniform(-5, 5)
        elif (is_northern and (month >= 6 and month <= 8)) or (not is_northern and (month >= 12 or month <= 2)):
            # Summer
            base_temp = 25 + random.uniform(-5, 5)
        else:
            # Winter
            base_temp = 5 + random.uniform(-5, 5)
        
        # Generate 7 days of 3-hourly forecasts (56 entries)
        for i in range(56):
            forecast_time = now + timedelta(hours=i*3)
            
            # Daily temperature cycle (coolest at night, warmest in afternoon)
            hour = forecast_time.hour
            temp_variation = -5 if hour < 6 else 5 if 12 <= hour <= 15 else 0
            
            # Add some randomness
            daily_random = random.uniform(-3, 3)
            
            temperature = base_temp + temp_variation + daily_random
            humidity = random.uniform(40, 90)
            
            # More rain probability in spring/summer
            rain_prob = 0.3 if ((is_northern and month >= 3 and month <= 8) or 
                               (not is_northern and (month >= 9 or month <= 2))) else 0.1
            
            rainfall = random.uniform(0, 10) if random.random() < rain_prob else 0
            
            forecast.append({
                'datetime': forecast_time,
                'temperature': temperature,
                'feels_like': temperature - random.uniform(0, 3),
                'temp_min': temperature - random.uniform(1, 3),
                'temp_max': temperature + random.uniform(1, 3),
                'humidity': humidity,
                'pressure': random.uniform(990, 1020),
                'rainfall': rainfall,
                'wind_speed': random.uniform(0, 15),
                'wind_direction': random.uniform(0, 360),
                'cloud_cover': random.uniform(0, 100),
                'weather_main': 'Rain' if rainfall > 0 else 'Clear' if random.random() > 0.3 else 'Clouds',
                'weather_description': 'light rain' if rainfall > 0 else 'clear sky' if random.random() > 0.3 else 'scattered clouds'
            })
        
        return forecast
        
    def build_alerts(self, forecast, crop_type):
        alerts = []

        frost = self.check_frost_risk(forecast, crop_type)
        logger.debug(f"Frost risk: {frost['risk_level']:.2f}")
        if frost['risk_level'] >= 0.3:
            alerts.append({**frost, 'type':'frost_warning'})

        drought = self.check_drought_risk(forecast)
        logger.debug(f"Drought risk: {drought['risk_level']:.2f}")
        if drought['risk_level'] >= 0.3:
            alerts.append({**drought, 'type':'drought_warning'})

        flood = self.check_flood_risk(forecast)
        logger.debug(f"Flood risk: {flood['risk_level']:.2f}")
        if flood['risk_level'] >= 0.3:
            alerts.append({**flood, 'type':'flood_warning'})

        disease = self.check_disease_risk(forecast, crop_type)
        logger.debug(f"Disease risk: {disease['risk_level']:.2f}")
        if disease['risk_level'] >= 0.3:
            alerts.append({**disease, 'type':'disease_warning'})

        return alerts


    def check_frost_risk(self, forecast, crop_type):
        frost_sensitive = {'Rice','Cotton','Sugarcane','Soybean','Maize'}
        if crop_type not in frost_sensitive:
            return {'risk_level':0,'severity':'low','message':'','recommended_action':''}
        # Count 3-hr periods next 24h with temp <2°C
        periods = [f for f in forecast[:8] if f.get('temperature') is not None and f['temperature']<2]
        count = len(periods)
        # 2 periods (6h) => medium, 3+ => high
        risk = min(count/4,1.0)
        sev = 'high' if count>=3 else 'medium' if count>=2 else 'low'
        msg = f"Frost risk: {count*3}h below 2°C expected today."
        action = "Use frost protection (mulch, heaters, sprinklers)."
        return {'risk_level':risk,'severity':sev,'message':msg,'recommended_action':action}

    def check_drought_risk(self, forecast):
        week = forecast[:56]  # ~7 days
        total_rain = sum(f.get('rainfall',0) for f in week)
        if total_rain < 25:
            risk, sev = 1.0, 'high'
        elif total_rain < 50:
            risk, sev = 0.6, 'medium'
        else:
            risk, sev = 0.2, 'low'
        msg = f"Weekly rainfall: {total_rain:.1f} mm"
        action = "Irrigate; prioritize water‐efficient methods."
        return {'risk_level':risk,'severity':sev,'message':msg,'recommended_action':action}

    def check_flood_risk(self, forecast):
        # Group by calendar day
        daily = {}
        for f in forecast[:56]:
            d = f['datetime'].date()
            daily.setdefault(d,0)
            daily[d] += f.get('rainfall',0)
        max_daily = max(daily.values()) if daily else 0
        total = sum(daily.values())
        # heavy rain thresholds
        if max_daily > 100 or total>200:
            risk, sev = 1.0, 'high'
        elif max_daily > 50:
            risk, sev = 0.6, 'medium'
        else:
            risk, sev = 0.2, 'low'
        msg = f"Max daily rain: {max_daily:.1f} mm; weekly total: {total:.1f} mm"
        action = "Ensure field drainage; avoid planting in flood zones."
        return {'risk_level':risk,'severity':sev,'message':msg,'recommended_action':action}

    def check_disease_risk(self, forecast, crop_type):
        # Count periods with RH>85% & 20–30°C
        count=0
        for f in forecast[:56]:
            if f.get('humidity',0)>85 and 20<=f.get('temperature',0)<=30:
                count+=1
        # Scale risk by count/8 per day
        days = min(count/8,7)
        risk = min(days/7,1.0)
        sev = 'high' if risk>0.6 else 'medium' if risk>0.3 else 'low'
        msg = f"{count*3}h high humidity & moderate temp (disease‐favorable)."
        action = "Scout for fungal disease; apply preventive treatments."
        return {'risk_level':risk,'severity':sev,'message':msg,'recommended_action':action}

    # ---------- Notifications & DB ----------
    def send_farmer_alerts(self, farmer_id: int, name: str, phone: Optional[str], alerts: List[Dict[str,str]]):
        for alert in alerts:
            self.save_alert_to_db(farmer_id, alert)

        # SMS (stub)
        sms = f"WEATHER ALERT for {name}:\n"
        for i, a in enumerate(alerts[:2]):
            sms += f"{i+1}. {a['message']}\nAction: {a['recommended_action']}\n"
            if 'data_source' in a:
                sms += f"Source: {a['data_source']}\n"
        self.send_sms(phone, sms)
        logger.info(f"Alerts saved & SMS queued for farmer_id={farmer_id}, alerts={len(alerts)}")

    def send_sms(self, phone_number: Optional[str], message: str):
        if not phone_number:
            logger.warning("No phone number; SMS not sent.")
            return
        # integrate with your provider here
        logger.info(f"SMS to {phone_number}: {message[:140]}...")

    def get_all_farmers(self):
        return self.conn.execute("SELECT farmer_id, name, latitude, longitude, land_size, crop_type, soil_type, phone_number, registration_date FROM farmers").fetchall()

    
    def save_alert_to_db(self, farmer_id: int, alert: Dict[str,str]):
        # Extract data source if available
        data_source = alert.get('data_source', 'unknown')
        
        self.conn.execute('''
            INSERT INTO weather_alerts 
            (farmer_id, alert_type, severity, message, recommended_action, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            farmer_id, alert['type'], alert['severity'],
            alert['message'], alert['recommended_action'], datetime.now()
        ))
        self.conn.commit()

    def list_recent_alerts(self, limit: int = 100):
        try:
            # Try to select with acknowledged column first
            cur = self.conn.execute('''
                SELECT id, farmer_id, alert_type, severity, message, recommended_action, created_at, acknowledged
                FROM weather_alerts
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        except sqlite3.OperationalError:
            # If acknowledged column doesn't exist, select without it
            cur = self.conn.execute('''
                SELECT id, farmer_id, alert_type, severity, message, recommended_action, created_at
                FROM weather_alerts
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        
        cols = [c[0] for c in cur.description]
        alerts = [dict(zip(cols, row)) for row in cur.fetchall()]
        
        # Add acknowledged=False for alerts that don't have this column
        for alert in alerts:
            if 'acknowledged' not in alert:
                alert['acknowledged'] = False
                
        return alerts
