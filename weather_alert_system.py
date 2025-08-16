import logging
import sqlite3
from datetime import datetime
import time
import threading
import requests
from typing import List, Dict, Any, Optional
from config import WEATHER_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, WEATHER_UNITS

# ---------- logging ----------
logger = logging.getLogger("weather_alerts")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

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

                forecast = self.get_detailed_forecast(lat, lon)
                if not forecast:
                    logger.warning(f"No forecast for farmer {farmer_id} '{name}'; skipping.")
                    continue

                alerts = self.build_alerts(forecast, crop_type)
                if alerts:
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
    def get_detailed_forecast(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Fetch 3-hourly forecast (up to ~5 days) and derive 7-day list (clips if API returns less)."""
        try:
            url = f"{WEATHER_API_BASE_URL}/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units={WEATHER_UNITS}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            lst = data.get("list", [])
            if not lst:
                return []

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
            return forecast

        except Exception as e:
            logger.exception(f"Forecast API error: {e}")
            return []

    def build_alerts(self, forecast: List[Dict[str, Any]], crop_type: str) -> List[Dict[str, str]]:
        alerts = []

        frost = self.check_frost_risk(forecast, crop_type)
        if frost['risk_level'] > 0.1:#0.7
            alerts.append({
                'type': 'frost_warning',
                'severity': 'high',
                'message': frost['message'],
                'recommended_action': frost['action']
            })

        drought = self.check_drought_risk(forecast)
        if drought['risk_level'] > 0.1:#0.6
            alerts.append({
                'type': 'drought_warning',
                'severity': 'medium',
                'message': drought['message'],
                'recommended_action': drought['action']
            })

        flood = self.check_flood_risk(forecast)
        if flood['risk_level'] > 0.1:#0.5
            alerts.append({
                'type': 'flood_warning',
                'severity': 'high',
                'message': flood['message'],
                'recommended_action': flood['action']
            })

        disease = self.check_disease_risk(forecast, crop_type)
        if disease['risk_level'] > 0.6:
            alerts.append({
                'type': 'disease_warning',
                'severity': 'medium',
                'message': disease['message'],
                'recommended_action': disease['action']
            })
        return alerts

    def check_frost_risk(self, forecast, crop_type):
        frost_sensitive = {'Rice', 'Cotton', 'Sugarcane', 'Soybean', 'Maize'}
        frost_threshold = 2.0
        if crop_type not in frost_sensitive:
            return {'risk_level': 0, 'message': '', 'action': ''}

        near_term = [f for f in forecast[:24] if f.get('temperature') is not None]
        if not near_term:
            return {'risk_level': 0, 'message': '', 'action': ''}

        frost_periods = [f['datetime'] for f in near_term if f['temperature'] < frost_threshold]
        if frost_periods:
            min_temp = min(f['temperature'] for f in near_term)
            risk = min(len(frost_periods) / 10.0, 1.0)
            return {
                'risk_level': risk,
                'message': f"FROST ALERT: Temperature down to {min_temp:.1f}°C expected in next 3 days.",
                'action': "Cover seedlings, use mulching/sprinklers, postpone irrigation before cold night."
            }
        return {'risk_level': 0, 'message': '', 'action': ''}

    def check_drought_risk(self, forecast):
        week = [f for f in forecast[:28] if 'rainfall' in f and 'humidity' in f]
        if not week:
            return {'risk_level': 0, 'message': '', 'action': ''}

        total_rain = sum(f['rainfall'] or 0 for f in week)
        avg_humidity = sum((f['humidity'] or 0) for f in week) / len(week)

        expected_rainfall = 25.0  # mm/week baseline
        score = 0.0
        if total_rain < expected_rainfall * 0.2: score = 0.9
        elif total_rain < expected_rainfall * 0.5: score = 0.6
        elif total_rain < expected_rainfall * 0.7: score = 0.3
        if avg_humidity < 40: score = min(score + 0.2, 1.0)

        if score > 0.1:#0.6
            return {
                'risk_level': score,
                'message': f"DROUGHT RISK: Only {total_rain:.1f} mm forecast in next 7 days.",
                'action': "Irrigate efficiently, prioritize critical growth stages, reduce evap loss with mulching."
            }
        return {'risk_level': score, 'message': '', 'action': ''}

    def check_flood_risk(self, forecast):
        week = forecast[:28]
        if not week:
            return {'risk_level': 0, 'message': '', 'action': ''}

        # group by day
        daily = {}
        for f in week:
            d = f['datetime'].date()
            daily.setdefault(d, 0.0)
            daily[d] += (f.get('rainfall') or 0.0)

        if not daily:
            return {'risk_level': 0, 'message': '', 'action': ''}

        daily_vals = list(daily.values())
        max_daily = max(daily_vals)
        total_weekly = sum(daily_vals)

        risk = 0.0
        if max_daily > 75: risk = 0.9
        elif max_daily > 50: risk = 0.6
        elif max_daily > 30: risk = 0.3
        if total_weekly > 200: risk = min(risk + 0.3, 1.0)

        if risk > 0.5:
            return {
                'risk_level': risk,
                'message': f"FLOOD RISK: Up to {max_daily:.1f} mm/day expected.",
                'action': "Clear drains, move inputs to higher ground, avoid field work during heavy rain."
            }
        return {'risk_level': risk, 'message': '', 'action': ''}

    def check_disease_risk(self, forecast, crop_type):
        week = forecast[:28]
        if not week:
            return {'risk_level': 0, 'message': '', 'action': ''}

        high_humid = sum(1 for f in week if (f.get('humidity') or 0) > 80)
        temps = [f.get('temperature') for f in week if f.get('temperature') is not None]
        rainy = sum(1 for f in week if (f.get('rainfall') or 0) > 2)

        risk = 0.0
        if high_humid > 10: risk += 0.4
        if temps and (max(temps) - min(temps) > 15): risk += 0.2
        if rainy > 15: risk += 0.3
        risk = min(risk, 1.0)

        crop_diseases = {
            'Rice': ['Blast', 'Sheath Blight'],
            'Wheat': ['Rust', 'Blight'],
            'Cotton': ['Bollworm', 'Wilt'],
            'Sugarcane': ['Red Rot', 'Smut'],
            'Soybean': ['Rust', 'Pod Borer'],
            'Maize': ['Borer', 'Rust']
        }
        if risk > 0.1:#0.6
            diseases = crop_diseases.get(crop_type, ['General diseases'])
            return {
                'risk_level': risk,
                'message': f"DISEASE RISK: Humid/rainy spell may trigger {', '.join(diseases[:2])}.",
                'action': "Scout fields; consider preventive spray per local advisories; improve field drainage."
            }
        return {'risk_level': risk, 'message': '', 'action': ''}

    # ---------- Notifications & DB ----------
    def send_farmer_alerts(self, farmer_id: int, name: str, phone: Optional[str], alerts: List[Dict[str,str]]):
        for alert in alerts:
            self.save_alert_to_db(farmer_id, alert)

        # SMS (stub)
        sms = f"WEATHER ALERT for {name}:\n"
        for i, a in enumerate(alerts[:2]):
            sms += f"{i+1}. {a['message']}\nAction: {a['recommended_action']}\n"
        self.send_sms(phone, sms)
        logger.info(f"Alerts saved & SMS queued for farmer_id={farmer_id}, alerts={len(alerts)}")

    def send_sms(self, phone_number: Optional[str], message: str):
        if not phone_number:
            logger.warning("No phone number; SMS not sent.")
            return
        # integrate with your provider here
        logger.info(f"SMS to {phone_number}: {message[:140]}...")

    def get_all_farmers(self):
        return self.conn.execute("SELECT * FROM farmers").fetchall()

    def save_alert_to_db(self, farmer_id: int, alert: Dict[str,str]):
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
        cur = self.conn.execute('''
            SELECT id, farmer_id, alert_type, severity, message, recommended_action, created_at, acknowledged
            FROM weather_alerts
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
