"""
================================================================================
  AgroScore v3.0 — Data Pipeline  (REFACTORED)
  Production-ready, API-integrated edition
================================================================================

API Sources
-----------
  1. OpenWeatherMap (WEATHER_API_KEY)  — current temperature + humidity
  2. Tomorrow.io   (TOMORROW_IO_API_KEY) — 7-day forecast: precipitation,
       frost risk, soil moisture, seasonal rainfall deviation
  3. NASA Earthdata AppEEARS (NASA_EARTHDATA_USERNAME / _PASSWORD)
       — MODIS MOD13Q1 NDVI + anomaly
  4. data.gov.in   (MARKET_API_KEY)    — mandi arrival prices
  5. RBI DBIE      (public, no key)    — repo rate + WPI inflation

Feature schema: 79 features (FEATURE_NAMES).  All API calls include
try/except for network errors, non-200 responses, rate-limit HTTP 429,
and empty JSON payloads.  A disk-backed SQLite cache avoids repeat calls.
================================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("agroscore_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_sh)
    _fh = logging.FileHandler("agroscore_pipeline.log", mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_fh)

# ── Config from .env ──────────────────────────────────────────────────────────
DATABASE_PATH         = os.getenv("DATABASE_PATH",           "agricred_data.db")
WEATHER_API_KEY       = os.getenv("WEATHER_API_KEY",         "")   # OpenWeatherMap
TOMORROW_IO_API_KEY   = os.getenv("TOMORROW_IO_API_KEY",     "")
NASA_EARTHDATA_USER   = os.getenv("NASA_EARTHDATA_USERNAME", "")
NASA_EARTHDATA_PASS   = os.getenv("NASA_EARTHDATA_PASSWORD", "")
MARKET_API_KEY        = os.getenv("MARKET_API_KEY",          "")   # data.gov.in
RBI_API_BASE          = os.getenv("RBI_API_BASE",            "https://dbie.rbi.org.in/api/v1")
REAL_LOAN_DATA_PATH   = os.getenv("REAL_LOAN_DATA_PATH",     "")
CACHE_ENABLED         = os.getenv("CACHE_ENABLED",           "true").lower() == "true"
CACHE_TTL             = int(os.getenv("CACHE_TTL",           "3600"))
RATE_LIMIT_CALLS      = int(os.getenv("RATE_LIMIT_CALLS",    "100"))
RATE_LIMIT_PERIOD     = int(os.getenv("RATE_LIMIT_PERIOD",   "3600"))
SYNTHETIC_BLEND_RATIO = float(os.getenv("SYNTHETIC_BLEND_RATIO", "0.4"))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FEATURE SCHEMA  (single source of truth — 79 features)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA: Dict[str, np.dtype] = {
    # Demographics
    "farmer_age":                           np.float64,
    "education_level":                      np.float64,
    "family_size":                          np.float64,
    # Land & crop
    "land_size":                            np.float64,
    "crop_type_encoded":                    np.float64,
    "irrigation_access":                    np.float64,
    # Weather (OpenWeatherMap + Tomorrow.io)
    "current_temperature":                  np.float64,
    "current_humidity":                     np.float64,
    "temperature_stress":                   np.float64,
    "humidity_stress":                      np.float64,
    "drought_risk_7days":                   np.float64,
    "frost_risk_7days":                     np.float64,
    "excess_rain_risk":                     np.float64,
    "soil_moisture_index":                  np.float64,   # Tomorrow.io
    # Satellite (NASA MODIS)
    "ndvi_current":                         np.float64,
    "ndvi_anomaly":                         np.float64,
    # Market (data.gov.in Agmarknet)
    "price_volatility":                     np.float64,
    "annual_income_proxy":                  np.float64,
    "current_price":                        np.float64,
    "market_demand_index":                  np.float64,
    "export_potential":                     np.float64,
    "storage_price_premium":                np.float64,
    "price_trend":                          np.float64,
    # Macro (RBI DBIE)
    "rbi_repo_rate":                        np.float64,
    "rbi_wpi_inflation":                    np.float64,
    # Financial ratios
    "loan_to_land_ratio":                   np.float64,
    "debt_to_income_ratio":                 np.float64,
    "payment_history_score":                np.float64,
    "savings_to_income_ratio":              np.float64,
    "credit_utilization":                   np.float64,
    "number_of_credit_sources":             np.float64,
    "informal_lending_dependency":          np.float64,
    # Log-transformed ratios (derived)
    "log_debt_to_income":                   np.float64,
    "log_loan_to_land":                     np.float64,
    "log_annual_income":                    np.float64,
    "log_mandi_distance":                   np.float64,
    # Yield & soil
    "yield_consistency":                    np.float64,
    "soil_health_index":                    np.float64,
    "nutrient_deficiency_risk":             np.float64,
    # Infrastructure
    "nearest_mandi_distance":               np.float64,
    "connectivity_index":                   np.float64,
    "road_quality_index":                   np.float64,
    "electricity_reliability":              np.float64,
    "mobile_network_strength":              np.float64,
    "bank_branch_distance":                 np.float64,
    "transport_cost_burden":                np.float64,
    "google_mandi_distance":                np.float64,
    # Support & social
    "insurance_coverage":                   np.float64,
    "cooperative_membership":               np.float64,
    "technology_adoption":                  np.float64,
    "diversification_index":                np.float64,
    "input_cost_index":                     np.float64,
    "mechanization_level":                  np.float64,
    "seed_quality_index":                   np.float64,
    "fertilizer_usage_efficiency":          np.float64,
    "pest_disease_risk":                    np.float64,
    "organic_farming_adoption":             np.float64,
    "precision_agriculture_usage":          np.float64,
    # Government / scheme
    "eligible_schemes_count":               np.float64,
    "subsidy_utilization":                  np.float64,
    "msp_eligibility":                      np.float64,
    "kisan_credit_card":                    np.float64,
    "government_training_participation":    np.float64,
    # Climate / seasonal
    "seasonal_rainfall_deviation":          np.float64,
    "historical_drought_frequency":         np.float64,
    "climate_change_vulnerability":         np.float64,
    # Community
    "community_leadership_role":            np.float64,
    "social_capital_index":                 np.float64,
    "extension_service_access":             np.float64,
    "peer_learning_participation":          np.float64,
    # Labor & supply
    "labor_availability":                   np.float64,
    "storage_access":                       np.float64,
    "supply_chain_integration":             np.float64,
    "disaster_preparedness":                np.float64,
    "alternative_income_sources":           np.float64,
    "livestock_ownership":                  np.float64,
    # Behavioral / interaction (derived)
    "seasonal_payment_consistency":         np.float64,
    "repayment_velocity_proxy":             np.float64,
    "climate_debt_compound_stress":         np.float64,
}

FEATURE_NAMES: List[str] = list(FEATURE_SCHEMA.keys())   # 79 ordered names
INPUT_DIM: int = len(FEATURE_NAMES)                       # 79 — model.py uses this

REAL_DATA_FEATURES = {
    "openweather":  {"current_temperature", "current_humidity"},
    "tomorrow_io":  {"drought_risk_7days", "frost_risk_7days", "excess_rain_risk",
                     "soil_moisture_index", "seasonal_rainfall_deviation"},
    "nasa_modis":   {"ndvi_current", "ndvi_anomaly"},
    "datagov":      {"current_price", "price_volatility", "price_trend"},
    "rbi":          {"rbi_repo_rate", "rbi_wpi_inflation"},
}

assert INPUT_DIM == 79, f"Feature count mismatch: expected 79, got {INPUT_DIM}"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FARMER FEATURE RECORD  (79 fields, typed dataclass)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FarmerFeatureRecord:
    """
    Complete 79-field feature record for one farmer at inference time.
    Defaults are calibrated Indian Kisan loan population medians.
    Call to_model_df() → model-ready single-row DataFrame (shape 1×79).
    """
    farmer_age:                         float = 42.0
    education_level:                    float = 2.0
    family_size:                        float = 4.0
    land_size:                          float = 2.5
    crop_type_encoded:                  float = 1.0
    irrigation_access:                  float = 0.0
    current_temperature:                float = 30.0
    current_humidity:                   float = 65.0
    temperature_stress:                 float = 0.13
    humidity_stress:                    float = 0.20
    drought_risk_7days:                 float = 0.30
    frost_risk_7days:                   float = 0.05
    excess_rain_risk:                   float = 0.10
    soil_moisture_index:                float = 0.40
    ndvi_current:                       float = 0.45
    ndvi_anomaly:                       float = 0.0
    price_volatility:                   float = 0.20
    annual_income_proxy:                float = 1.10
    current_price:                      float = 20000.0
    market_demand_index:                float = 0.45
    export_potential:                   float = 0.25
    storage_price_premium:              float = 0.10
    price_trend:                        float = 0.02
    rbi_repo_rate:                      float = 6.50
    rbi_wpi_inflation:                  float = 4.50
    loan_to_land_ratio:                 float = 0.25
    debt_to_income_ratio:               float = 0.40
    payment_history_score:              float = 0.75
    savings_to_income_ratio:            float = 0.10
    credit_utilization:                 float = 0.45
    number_of_credit_sources:           float = 1.5
    informal_lending_dependency:        float = 0.25
    log_debt_to_income:                 float = 0.0
    log_loan_to_land:                   float = 0.0
    log_annual_income:                  float = 0.0
    log_mandi_distance:                 float = 0.0
    yield_consistency:                  float = 0.65
    soil_health_index:                  float = 0.60
    nutrient_deficiency_risk:           float = 0.35
    nearest_mandi_distance:             float = 15.0
    connectivity_index:                 float = 0.55
    road_quality_index:                 float = 0.55
    electricity_reliability:            float = 0.70
    mobile_network_strength:            float = 0.80
    bank_branch_distance:               float = 12.0
    transport_cost_burden:              float = 0.30
    google_mandi_distance:              float = 15.0
    insurance_coverage:                 float = 0.0
    cooperative_membership:             float = 0.0
    technology_adoption:                float = 0.40
    diversification_index:              float = 0.30
    input_cost_index:                   float = 0.55
    mechanization_level:                float = 0.35
    seed_quality_index:                 float = 0.65
    fertilizer_usage_efficiency:        float = 0.60
    pest_disease_risk:                  float = 0.20
    organic_farming_adoption:           float = 0.15
    precision_agriculture_usage:        float = 0.20
    eligible_schemes_count:             float = 2.0
    subsidy_utilization:                float = 0.35
    msp_eligibility:                    float = 1.0
    kisan_credit_card:                  float = 0.0
    government_training_participation:  float = 0.15
    seasonal_rainfall_deviation:        float = 0.0
    historical_drought_frequency:       float = 1.5
    climate_change_vulnerability:       float = 0.25
    community_leadership_role:          float = 0.0
    social_capital_index:               float = 0.45
    extension_service_access:           float = 0.40
    peer_learning_participation:        float = 0.30
    labor_availability:                 float = 0.65
    storage_access:                     float = 0.0
    supply_chain_integration:           float = 0.30
    disaster_preparedness:              float = 0.35
    alternative_income_sources:         float = 0.40
    livestock_ownership:                float = 0.0
    seasonal_payment_consistency:       float = 0.0
    repayment_velocity_proxy:           float = 0.0
    climate_debt_compound_stress:       float = 0.0

    def _compute_derived(self) -> None:
        """Recompute log-transforms and interaction terms in-place."""
        self.log_debt_to_income = float(np.log1p(self.debt_to_income_ratio))
        self.log_loan_to_land   = float(np.log1p(self.loan_to_land_ratio))
        self.log_annual_income  = float(np.log1p(self.annual_income_proxy))
        self.log_mandi_distance = float(np.log1p(self.nearest_mandi_distance))

        self.seasonal_payment_consistency = float(np.clip(
            self.payment_history_score * 0.7
            + (1.0 - self.price_volatility) * 0.3, 0, 1
        ))
        dti_safe = max(self.debt_to_income_ratio, 0.05)
        self.repayment_velocity_proxy = float(np.clip(
            self.payment_history_score / dti_safe * 0.7, 0, 1
        ))
        # NDVI-augmented compound stress (novel feature)
        ndvi_weight = float(np.clip(1.0 - self.ndvi_current, 0.2, 1.0))
        self.climate_debt_compound_stress = float(np.clip(
            math.sqrt(
                self.drought_risk_7days
                * self.debt_to_income_ratio
                * ndvi_weight
                * (1.0 + self.rbi_wpi_inflation / 20.0)
            ), 0, 1
        ))

    def to_model_df(self) -> pd.DataFrame:
        """
        Returns a 1×79 DataFrame aligned to FEATURE_NAMES.
        Derived features are recomputed fresh each call.
        """
        self._compute_derived()
        row = {k: getattr(self, k) for k in FEATURE_NAMES}
        return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATA QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    openweather_imputed:  bool = False
    tomorrow_imputed:     bool = False
    ndvi_imputed:         bool = False
    market_imputed:       bool = False
    rbi_imputed:          bool = False
    loan_imputed:         bool = False
    imputation_notes:     List[str] = field(default_factory=list)
    source_provenance:    Dict[str, str] = field(default_factory=dict)

    @property
    def overall_confidence(self) -> str:
        imputed_count = sum([
            self.openweather_imputed, self.tomorrow_imputed, self.ndvi_imputed,
            self.market_imputed, self.rbi_imputed, self.loan_imputed
        ])
        if imputed_count == 0: return "HIGH"
        if imputed_count <= 2: return "MEDIUM"
        return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SQLITE CACHE + DATABASE
# ══════════════════════════════════════════════════════════════════════════════

class AgroScoreDatabase:
    """Thin SQLite wrapper — farmers, loans, and an API response cache."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS farmers (
        farmer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, state TEXT, district TEXT,
        latitude REAL, longitude REAL, land_size REAL,
        crop_type TEXT, irrigation_access INTEGER,
        education_level INTEGER, family_size INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS loans (
        loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER, amount REAL, interest_rate REAL,
        duration_months INTEGER, disbursed_date TEXT, due_date TEXT,
        status TEXT, repaid_amount REAL, credit_score INTEGER, risk_level TEXT,
        FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id)
    );
    CREATE TABLE IF NOT EXISTS api_cache (
        cache_key TEXT PRIMARY KEY,
        source TEXT, payload TEXT,
        cached_at REAL, ttl INTEGER
    );
    CREATE TABLE IF NOT EXISTS portfolio_metrics (
        date TEXT PRIMARY KEY, total_farmers INTEGER, total_loans INTEGER,
        total_portfolio_value REAL, active_loans INTEGER, repaid_loans INTEGER,
        defaulted_loans INTEGER, default_rate REAL, avg_credit_score REAL,
        total_land_size REAL, avg_loan_amount REAL
    );
    """

    def __init__(self, path: str = DATABASE_PATH):
        self.path = path
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Cache helpers ─────────────────────────────────────────────────────────
    def get_cache(self, key: str) -> Optional[dict]:
        if not CACHE_ENABLED:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT payload, cached_at, ttl FROM api_cache WHERE cache_key=?",
                    (key,)
                ).fetchone()
            if row and (time.time() - row["cached_at"]) < row["ttl"]:
                return json.loads(row["payload"])
        except Exception as e:
            logger.debug(f"Cache read error ({key}): {e}")
        return None

    def set_cache(self, key: str, source: str, data: dict, ttl: int = CACHE_TTL) -> None:
        if not CACHE_ENABLED:
            return
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO api_cache
                       (cache_key, source, payload, cached_at, ttl)
                       VALUES (?,?,?,?,?)""",
                    (key, source, json.dumps(data), time.time(), ttl)
                )
        except Exception as e:
            logger.debug(f"Cache write error ({key}): {e}")

    # ── Farmer helpers ────────────────────────────────────────────────────────
    def get_farmer(self, farmer_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM farmers WHERE farmer_id=?", (farmer_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_loans(self, farmer_id: int) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM loans WHERE farmer_id=?", (farmer_id,)
            ).fetchall()
        return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class _RateLimiter:
    """Simple token-bucket rate limiter shared across API clients."""
    def __init__(self, calls: int = RATE_LIMIT_CALLS, period: int = RATE_LIMIT_PERIOD):
        self._calls  = calls
        self._period = period
        self._count  = 0
        self._reset  = time.time() + period

    def check(self) -> bool:
        now = time.time()
        if now > self._reset:
            self._count = 0
            self._reset = now + self._period
        if self._count < self._calls:
            self._count += 1
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — REAL-DATA API CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

class OpenWeatherClient:
    """
    OpenWeatherMap Current Weather API.
    Key  : WEATHER_API_KEY (env var, maps to .env WEATHER_API_KEY)
    Docs : https://openweathermap.org/current
    Free tier: 1,000 calls/day, 60/min.
    Returns: temperature (°C), humidity (%)
    """
    BASE = "https://api.openweathermap.org/data/2.5/weather"
    _rl  = _RateLimiter(60, 60)

    REGION_DEFAULTS = {
        "Punjab":      (28.0, 55.0),
        "Maharashtra": (32.0, 65.0),
        "UP":          (30.0, 70.0),
        "Karnataka":   (29.0, 62.0),
        "AP":          (33.0, 68.0),
        "WB":          (31.0, 78.0),
        "Gujarat":     (34.0, 58.0),
        "MP":          (31.0, 60.0),
        "_default":    (30.0, 65.0),
    }

    def __init__(self, api_key: str = WEATHER_API_KEY, db: Optional[AgroScoreDatabase] = None):
        self.key = api_key
        self.db  = db

    def get_current(self, lat: float, lon: float,
                    region: str = "_default") -> Tuple[dict, bool]:
        """
        Returns (result_dict, was_imputed).
        result_dict keys: temperature (°C), humidity (%)
        """
        if not self.key:
            logger.warning("OpenWeatherMap key missing — using fallback.")
            return self._fallback(region), True

        cache_key = f"ow_{lat:.3f}_{lon:.3f}"
        if self.db:
            cached = self.db.get_cache(cache_key)
            if cached:
                return cached, False

        if not self._rl.check():
            logger.warning("OpenWeatherMap rate limit hit — using fallback.")
            return self._fallback(region), True

        try:
            resp = requests.get(
                self.BASE,
                params={"lat": lat, "lon": lon, "appid": self.key, "units": "metric"},
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning("OpenWeatherMap HTTP 429 (rate limited).")
                return self._fallback(region), True
            if resp.status_code != 200:
                logger.warning(f"OpenWeatherMap status {resp.status_code}.")
                return self._fallback(region), True
            payload = resp.json()
            if "main" not in payload:
                logger.warning("OpenWeatherMap: empty 'main' in response.")
                return self._fallback(region), True
            result = {
                "temperature": float(payload["main"]["temp"]),
                "humidity":    float(payload["main"]["humidity"]),
            }
            if self.db:
                self.db.set_cache(cache_key, "openweather", result, 1800)
            return result, False
        except requests.exceptions.Timeout:
            logger.warning("OpenWeatherMap request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"OpenWeatherMap connection error: {e}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"OpenWeatherMap parse error: {e}")
        except Exception as e:
            logger.warning(f"OpenWeatherMap unexpected error: {e}")
        return self._fallback(region), True

    def _fallback(self, region: str) -> dict:
        t, h = self.REGION_DEFAULTS.get(region, self.REGION_DEFAULTS["_default"])
        return {"temperature": t, "humidity": h}


class TomorrowIOClient:
    """
    Tomorrow.io Timelines API v4 — 7-day hyperlocal forecast.
    Key  : TOMORROW_IO_API_KEY
    Docs : https://docs.tomorrow.io/reference/post-timelines
    Free tier: 500 calls/day, 25/hour.

    Extracts:
        precipitationProbability  → drought_risk / excess_rain_risk
        freezingRainIntensity     → frost_risk_7days
        soilMoistureVolumetric0To10 → soil_moisture_index
        precipitationIntensity    → seasonal_rainfall_deviation
    """
    BASE = "https://api.tomorrow.io/v4/timelines"
    _rl  = _RateLimiter(25, 3600)

    def __init__(self, api_key: str = TOMORROW_IO_API_KEY,
                 db: Optional[AgroScoreDatabase] = None):
        self.key = api_key
        self.db  = db

    def get_forecast(self, lat: float, lon: float) -> Tuple[dict, bool]:
        """
        Returns (result_dict, was_imputed).
        result_dict keys: drought_risk_7days, frost_risk_7days, excess_rain_risk,
                          soil_moisture_index, seasonal_rainfall_deviation
        """
        if not self.key:
            logger.warning("Tomorrow.io key missing — using fallback.")
            return self._fallback(), True

        cache_key = f"tmr_{lat:.3f}_{lon:.3f}"
        if self.db:
            cached = self.db.get_cache(cache_key)
            if cached:
                return cached, False

        if not self._rl.check():
            logger.warning("Tomorrow.io rate limit hit — using fallback.")
            return self._fallback(), True

        try:
            body = {
                "location":  f"{lat},{lon}",
                "fields":    [
                    "precipitationProbability",
                    "precipitationIntensity",
                    "freezingRainIntensity",
                    "soilMoistureVolumetric0To10",
                    "temperature",
                ],
                "units":     "metric",
                "timesteps": ["1d"],
                "startTime": "now",
                "endTime":   "nowPlus7d",
            }
            resp = requests.post(
                self.BASE,
                json=body,
                headers={"apikey": self.key, "content-type": "application/json"},
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("Tomorrow.io HTTP 429 (rate limited).")
                return self._fallback(), True
            if resp.status_code != 200:
                logger.warning(f"Tomorrow.io status {resp.status_code}: {resp.text[:200]}")
                return self._fallback(), True

            payload = resp.json()
            try:
                intervals = payload["data"]["timelines"][0]["intervals"]
            except (KeyError, IndexError):
                logger.warning("Tomorrow.io: unexpected response shape.")
                return self._fallback(), True

            if not intervals:
                logger.warning("Tomorrow.io: zero intervals returned.")
                return self._fallback(), True

            precip_probs   = [i["values"].get("precipitationProbability", 50)  for i in intervals]
            rain_intensity = [i["values"].get("precipitationIntensity",    0)  for i in intervals]
            freeze_vals    = [i["values"].get("freezingRainIntensity",      0)  for i in intervals]
            sm_vals        = [i["values"].get("soilMoistureVolumetric0To10", 0.3) for i in intervals]
            temps          = [i["values"].get("temperature",               28)  for i in intervals]

            avg_precip_prob = float(np.mean(precip_probs)) / 100.0
            drought_risk    = float(np.clip(1.0 - avg_precip_prob, 0, 1))
            max_rain        = float(max(rain_intensity))
            excess_rain     = 0.70 if max_rain > 50 else (0.30 if max_rain > 25 else 0.05)
            min_temp        = float(min(temps))
            frost_risk      = float(np.clip((2.0 - min_temp) / 10.0, 0, 1))
            soil_moisture   = float(np.clip(np.mean(sm_vals) / 0.5, 0, 1))
            # Deviation from India climatological baseline of ~30 mm/week
            total_rain_mm   = sum(rain_intensity) * 24 * 7
            deviation       = float(np.clip((total_rain_mm - 30.0) / 30.0, -1, 1))

            result = {
                "drought_risk_7days":         drought_risk,
                "frost_risk_7days":           frost_risk,
                "excess_rain_risk":           excess_rain,
                "soil_moisture_index":        soil_moisture,
                "seasonal_rainfall_deviation": deviation,
            }
            if self.db:
                self.db.set_cache(cache_key, "tomorrow_io", result, 3600)
            return result, False

        except requests.exceptions.Timeout:
            logger.warning("Tomorrow.io request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Tomorrow.io connection error: {e}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Tomorrow.io parse error: {e}")
        except Exception as e:
            logger.warning(f"Tomorrow.io unexpected error: {e}")
        return self._fallback(), True

    def _fallback(self) -> dict:
        return {
            "drought_risk_7days":          0.30,
            "frost_risk_7days":            0.05,
            "excess_rain_risk":            0.10,
            "soil_moisture_index":         0.40,
            "seasonal_rainfall_deviation": 0.0,
        }


class NASAEarthdataClient:
    """
    NASA Earthdata AppEEARS point query — MODIS MOD13Q1 NDVI.
    Creds: NASA_EARTHDATA_USERNAME / NASA_EARTHDATA_PASSWORD
    Docs : https://appeears.earthdatacloud.nasa.gov/api/
    Product: MOD13Q1.061 layer _250m_16_days_NDVI (scale factor 0.0001)
    No per-call cost; credentials required for bulk download.
    """
    BASE = "https://appeears.earthdatacloud.nasa.gov/api"

    def __init__(self, username: str = NASA_EARTHDATA_USER,
                 password: str = NASA_EARTHDATA_PASS,
                 db: Optional[AgroScoreDatabase] = None):
        self.username = username
        self.password = password
        self.db       = db
        self._token:  Optional[str] = None
        self._token_expiry = datetime.min

    # ── Authentication ────────────────────────────────────────────────────────
    def _get_token(self) -> Optional[str]:
        if self._token and datetime.now() < self._token_expiry:
            return self._token
        if not self.username or not self.password:
            return None
        try:
            resp = requests.post(
                f"{self.BASE}/login",
                auth=(self.username, self.password),
                timeout=15,
            )
            if resp.status_code == 200:
                self._token        = resp.json()["token"]
                self._token_expiry = datetime.now() + timedelta(hours=47)
                return self._token
            logger.warning(f"NASA Earthdata login failed: status {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("NASA Earthdata login timed out.")
        except Exception as e:
            logger.warning(f"NASA Earthdata login error: {e}")
        return None

    # ── NDVI fetch ────────────────────────────────────────────────────────────
    def get_ndvi(self, lat: float, lon: float) -> Tuple[dict, bool]:
        """
        Returns (result_dict, was_imputed).
        result_dict keys: ndvi_current (MODIS scaled, –1..1), ndvi_anomaly (z-score ±2)
        """
        cache_key = f"ndvi_{lat:.3f}_{lon:.3f}"
        if self.db:
            cached = self.db.get_cache(cache_key)
            if cached:
                return cached, False

        token = self._get_token()
        if not token:
            logger.warning("NASA Earthdata: no token — using regional fallback.")
            return self._fallback(lat), True

        try:
            headers    = {"Authorization": f"Bearer {token}"}
            end_dt     = datetime.now()
            start_dt   = end_dt - timedelta(days=16)
            payload = {
                "task_type": "point",
                "task_name": f"ndvi_{lat:.3f}_{lon:.3f}_{int(time.time())}",
                "params": {
                    "dates": [{
                        "startDate": start_dt.strftime("%m-%d-%Y"),
                        "endDate":   end_dt.strftime("%m-%d-%Y"),
                    }],
                    "layers": [{"product": "MOD13Q1.061",
                                "layer":   "_250m_16_days_NDVI"}],
                    "coordinates": [{
                        "longitude": lon, "latitude": lat,
                        "id": "farm", "category": "farm",
                    }],
                    "output": {"format": {"type": "geotiff"}, "projection": "native"},
                },
            }
            task_resp = requests.post(
                f"{self.BASE}/task", json=payload, headers=headers, timeout=20
            )
            if task_resp.status_code != 202:
                logger.warning(f"NASA AppEEARS task submit status {task_resp.status_code}")
                return self._fallback(lat), True

            task_id = task_resp.json().get("task_id")
            if not task_id:
                logger.warning("NASA AppEEARS: no task_id in response.")
                return self._fallback(lat), True

            # Poll up to 30 s (6 × 5 s) — typical for a single-point query
            for _ in range(6):
                time.sleep(5)
                try:
                    status_resp = requests.get(
                        f"{self.BASE}/task/{task_id}", headers=headers, timeout=10
                    )
                    if status_resp.status_code == 200:
                        status = status_resp.json().get("status", "")
                        if status == "done":
                            break
                        if status in ("error", "deleted"):
                            logger.warning(f"NASA AppEEARS task status: {status}")
                            return self._fallback(lat), True
                except Exception as e:
                    logger.debug(f"NASA poll error: {e}")
            else:
                logger.warning("NASA AppEEARS task did not complete in 30 s.")
                return self._fallback(lat), True

            # Download result CSV
            files_resp = requests.get(
                f"{self.BASE}/bundle/{task_id}", headers=headers, timeout=10
            )
            if files_resp.status_code != 200:
                return self._fallback(lat), True

            for f_info in files_resp.json().get("files", []):
                if not f_info.get("file_name", "").endswith(".csv"):
                    continue
                csv_resp = requests.get(
                    f"{self.BASE}/bundle/{task_id}/{f_info['file_id']}",
                    headers=headers, stream=True, timeout=15,
                )
                if csv_resp.status_code != 200:
                    continue
                lines = csv_resp.text.strip().split("\n")
                if len(lines) < 2:
                    continue
                cols = lines[0].split(",")
                vals = lines[-1].split(",")
                ndvi_col = "_250m_16_days_NDVI"
                if ndvi_col not in cols:
                    logger.warning(f"NASA CSV missing column '{ndvi_col}'.")
                    continue
                try:
                    raw_ndvi     = float(vals[cols.index(ndvi_col)])
                    ndvi_scaled  = raw_ndvi * 0.0001   # MODIS scale factor
                    ndvi_current = float(np.clip(ndvi_scaled, -1.0, 1.0))
                    baseline     = self._regional_ndvi_baseline(lat)
                    ndvi_anomaly = float(np.clip((ndvi_current - baseline) / 0.15, -2.0, 2.0))
                    result = {"ndvi_current": ndvi_current, "ndvi_anomaly": ndvi_anomaly}
                    if self.db:
                        self.db.set_cache(cache_key, "nasa_modis", result, 86400)
                    return result, False
                except (ValueError, IndexError) as e:
                    logger.warning(f"NASA NDVI value parse error: {e}")

        except requests.exceptions.Timeout:
            logger.warning("NASA Earthdata request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"NASA Earthdata connection error: {e}")
        except Exception as e:
            logger.warning(f"NASA Earthdata unexpected error: {e}")
        return self._fallback(lat), True

    def _regional_ndvi_baseline(self, lat: float) -> float:
        """5-year MODIS MOD13Q1 climatological mean by Indian latitude band."""
        if lat > 28:   return 0.38   # NW India (Punjab, N-UP)
        if lat > 22:   return 0.45   # Central (MP, Maharashtra)
        if lat > 16:   return 0.52   # Deccan plateau (Karnataka, AP)
        return 0.58                   # South India

    def _fallback(self, lat: float) -> dict:
        return {
            "ndvi_current": self._regional_ndvi_baseline(lat),
            "ndvi_anomaly": 0.0,
        }


class DataGovMarketClient:
    """
    data.gov.in Agmarknet mandi arrival prices.
    Key      : MARKET_API_KEY
    Resource : 9ef84268-d588-465a-a308-a864a43d0070
    Docs     : https://data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
    Free tier: 50 calls/hour assumed; register at https://data.gov.in
    Returns: current_price (₹/quintal), price_volatility, price_trend
    """
    RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
    API_BASE    = "https://api.data.gov.in/resource"
    _rl         = _RateLimiter(50, 3600)

    VOLATILITY_MAP = {
        "Rice": 0.18, "Wheat": 0.12, "Cotton": 0.35,
        "Sugarcane": 0.25, "Soybean": 0.28, "Maize": 0.20,
    }
    BASE_PRICES = {
        "Rice": 2100.0, "Wheat": 2015.0, "Cotton": 6200.0,
        "Sugarcane": 315.0, "Soybean": 4000.0, "Maize": 1750.0,
    }
    CROP_ALIASES = {
        "rice": "Rice", "wheat": "Wheat", "cotton": "Cotton",
        "sugarcane": "Sugarcane", "soybean": "Soybean", "maize": "Maize",
    }

    def __init__(self, api_key: str = MARKET_API_KEY,
                 db: Optional[AgroScoreDatabase] = None):
        self.key = api_key
        self.db  = db

    def get_price(self, crop: str, state: str = "") -> Tuple[dict, bool]:
        """
        Returns (result_dict, was_imputed).
        result_dict keys: current_price, price_volatility, price_trend, source
        """
        crop_std  = self.CROP_ALIASES.get(crop.lower(), crop.title())
        cache_key = f"mkt_{crop_std}_{state}"
        if self.db:
            cached = self.db.get_cache(cache_key)
            if cached:
                return cached, False

        if not self.key:
            logger.warning("data.gov.in key missing — using fallback.")
            return self._fallback(crop_std), True

        if not self._rl.check():
            logger.warning("data.gov.in rate limit hit — using fallback.")
            return self._fallback(crop_std), True

        try:
            params = {
                "api-key":            self.key,
                "format":             "json",
                "limit":              5,
                "filters[commodity]": crop_std,
            }
            if state:
                params["filters[state]"] = state.title()

            resp = requests.get(
                f"{self.API_BASE}/{self.RESOURCE_ID}",
                params=params,
                headers={"User-Agent": "AgroScore/3.0", "Accept": "application/json"},
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("data.gov.in HTTP 429 (rate limited).")
                return self._fallback(crop_std), True
            if resp.status_code != 200:
                logger.warning(f"data.gov.in status {resp.status_code}.")
                return self._fallback(crop_std), True

            data    = resp.json()
            records = data.get("records") or data.get("data") or []

            if not records:
                logger.warning(f"data.gov.in: no records for crop={crop_std}, state={state}.")
                return self._fallback(crop_std), True

            modal = float(records[0].get("modal_price", records[0].get("modalPrice", 0) or 0))
            if modal <= 0:
                return self._fallback(crop_std), True

            prices = [float(r.get("modal_price", modal) or modal) for r in records]
            trend  = float(np.clip((prices[0] - prices[-1]) / (prices[-1] + 1), -0.3, 0.3))

            result = {
                "current_price":    modal,
                "price_volatility": self.VOLATILITY_MAP.get(crop_std, 0.20),
                "price_trend":      trend,
                "source":           "data.gov.in",
            }
            if self.db:
                self.db.set_cache(cache_key, "datagov", result, 3600)
            return result, False

        except requests.exceptions.Timeout:
            logger.warning("data.gov.in request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"data.gov.in connection error: {e}")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"data.gov.in parse error: {e}")
        except Exception as e:
            logger.warning(f"data.gov.in unexpected error: {e}")
        return self._fallback(crop_std), True

    def _fallback(self, crop: str) -> dict:
        return {
            "current_price":    self.BASE_PRICES.get(crop, 2000.0),
            "price_volatility": self.VOLATILITY_MAP.get(crop, 0.20),
            "price_trend":      0.02,
            "source":           "fallback",
        }


class RBIClient:
    """
    Reserve Bank of India DBIE REST API — macro indicators.
    No API key required.  Public endpoint:
      https://dbie.rbi.org.in/api/v1/getSeriesData
    Series used:
      FMRR  — Policy repo rate (daily)
      CMPIE — WPI All Commodities YoY (monthly)
    Fallback: 6.50 % repo, 4.50 % WPI.
    """
    REPO_SERIES = "FMRR"
    WPI_SERIES  = "CMPIE"

    def __init__(self, base: str = RBI_API_BASE,
                 db: Optional[AgroScoreDatabase] = None):
        self.base = base
        self.db   = db

    def get_macro(self) -> Tuple[dict, bool]:
        """Returns (result_dict, was_imputed). Keys: rbi_repo_rate, rbi_wpi_inflation."""
        cache_key = "rbi_macro"
        if self.db:
            cached = self.db.get_cache(cache_key)
            if cached:
                return cached, False

        repo_rate = 6.50
        wpi       = 4.50
        imputed   = True

        try:
            resp = requests.get(
                f"{self.base}/getSeriesData",
                params={"seriesId": self.REPO_SERIES, "frequency": "D", "limit": 1},
                timeout=15,
            )
            if resp.status_code == 200:
                series = resp.json().get("data", [])
                if series:
                    repo_rate = float(series[0].get("value", repo_rate))
                    imputed   = False
            else:
                logger.warning(f"RBI DBIE repo rate status {resp.status_code}.")
        except Exception as e:
            logger.warning(f"RBI DBIE repo rate fetch error: {e}")

        try:
            resp2 = requests.get(
                f"{self.base}/getSeriesData",
                params={"seriesId": self.WPI_SERIES, "frequency": "M", "limit": 1},
                timeout=15,
            )
            if resp2.status_code == 200:
                wpi_data = resp2.json().get("data", [])
                if wpi_data:
                    wpi = float(wpi_data[0].get("value", wpi))
        except Exception as e:
            logger.warning(f"RBI DBIE WPI fetch error: {e}")

        result = {"rbi_repo_rate": repo_rate, "rbi_wpi_inflation": wpi}
        if self.db:
            self.db.set_cache(cache_key, "rbi", result, 86400)
        return result, imputed


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — SOIL HEALTH COMPUTER
# ══════════════════════════════════════════════════════════════════════════════

class SoilHealthComputer:
    """
    Converts raw Soil Health Card parameters to composite model features.
    Weights sourced from ICAR soil health guidelines.
    """
    def compute_index(self, ph: float, nitrogen: float, phosphorus: float,
                      potassium: float, organic_carbon: float) -> Tuple[float, float]:
        ph_score  = float(np.clip(1.0 - ((ph - 7.0) / 1.5) ** 2, 0.0, 1.0))
        n_score   = float(np.clip(nitrogen       / 60.0,   0.0, 1.0))
        p_score   = float(np.clip(phosphorus     / 10.0,   0.0, 1.0))
        k_score   = float(np.clip(potassium      / 120.0,  0.0, 1.0))
        oc_score  = float(np.clip(organic_carbon / 1.5,    0.0, 1.0))
        shi = (ph_score * 0.25 + n_score * 0.20 + p_score * 0.15
               + k_score * 0.15 + oc_score * 0.25)
        ndr = float(np.clip(1.0 - (n_score + p_score + k_score) / 3.0, 0.0, 1.0))
        return float(np.clip(shi, 0.0, 1.0)), ndr

    def get_from_district(self, district: str) -> Tuple[dict, bool]:
        """Placeholder — returns population-average soil parameters."""
        return {
            "ph_level":       7.1,
            "nitrogen":       48.0,
            "phosphorus":      6.2,
            "potassium":      75.0,
            "organic_carbon":  0.91,
        }, True


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PIPELINE FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class AgroScorePipelineFeatureBuilder:
    """
    Fetches all real-data signals, assembles them into a FarmerFeatureRecord,
    and returns it together with a DataQualityReport.
    """

    CROP_MAP = {1: "Rice", 2: "Wheat", 3: "Cotton", 4: "Sugarcane",
                5: "Soybean", 6: "Maize"}

    def __init__(self, db: AgroScoreDatabase,
                 ow: OpenWeatherClient,
                 tmr: TomorrowIOClient,
                 nasa: NASAEarthdataClient,
                 mkt: DataGovMarketClient,
                 rbi: RBIClient,
                 soil: SoilHealthComputer):
        self.db   = db
        self.ow   = ow
        self.tmr  = tmr
        self.nasa = nasa
        self.mkt  = mkt
        self.rbi  = rbi
        self.soil = soil

    def build(self, farmer_id: int) -> Tuple[FarmerFeatureRecord, DataQualityReport]:
        farmer = self.db.get_farmer(farmer_id)
        if not farmer:
            raise ValueError(f"farmer_id={farmer_id} not found in database.")
        loans  = self.db.get_loans(farmer_id)
        q      = DataQualityReport()

        lat   = float(farmer.get("latitude",  20.5))
        lon   = float(farmer.get("longitude", 78.9))
        state = str(farmer.get("state", "_default"))
        crop  = self.CROP_MAP.get(int(farmer.get("crop_type", 1) or 1), "Rice")

        rec = FarmerFeatureRecord(
            farmer_age         = float(farmer.get("farmer_age",       42.0) or 42.0),
            education_level    = float(farmer.get("education_level",   2.0) or 2.0),
            family_size        = float(farmer.get("family_size",       4.0) or 4.0),
            land_size          = float(farmer.get("land_size",         2.5) or 2.5),
            crop_type_encoded  = float(farmer.get("crop_type",         1.0) or 1.0),
            irrigation_access  = float(farmer.get("irrigation_access", 0.0) or 0.0),
        )

        # ── 1. OpenWeatherMap ──────────────────────────────────────────────────
        ow_data, q.openweather_imputed = self.ow.get_current(lat, lon, state)
        rec.current_temperature = ow_data["temperature"]
        rec.current_humidity    = ow_data["humidity"]
        rec.temperature_stress  = float(np.clip((rec.current_temperature - 28) / 15, 0, 1))
        rec.humidity_stress     = float(np.clip(abs(rec.current_humidity - 60) / 25, 0, 1))
        q.source_provenance["weather_current"] = (
            "openweather_api" if not q.openweather_imputed else "regional_fallback"
        )

        # ── 2. Tomorrow.io ─────────────────────────────────────────────────────
        tmr_data, q.tomorrow_imputed = self.tmr.get_forecast(lat, lon)
        rec.drought_risk_7days          = tmr_data["drought_risk_7days"]
        rec.frost_risk_7days            = tmr_data["frost_risk_7days"]
        rec.excess_rain_risk            = tmr_data["excess_rain_risk"]
        rec.soil_moisture_index         = tmr_data["soil_moisture_index"]
        rec.seasonal_rainfall_deviation = tmr_data["seasonal_rainfall_deviation"]
        q.source_provenance["weather_forecast"] = (
            "tomorrow_io_api" if not q.tomorrow_imputed else "climatological_fallback"
        )

        # ── 3. NASA MODIS NDVI ─────────────────────────────────────────────────
        ndvi_data, q.ndvi_imputed = self.nasa.get_ndvi(lat, lon)
        rec.ndvi_current  = ndvi_data["ndvi_current"]
        rec.ndvi_anomaly  = ndvi_data["ndvi_anomaly"]
        # NDVI feeds yield_consistency proxy
        rec.yield_consistency = float(np.clip(
            0.4 + rec.ndvi_current * 0.4
            + rec.irrigation_access * 0.2
            - max(rec.ndvi_anomaly, 0) * 0.1, 0.2, 1.0
        ))
        q.source_provenance["ndvi"] = (
            "nasa_modis_api" if not q.ndvi_imputed else "regional_ndvi_baseline"
        )

        # ── 4. data.gov.in Agmarknet ───────────────────────────────────────────
        mkt_data, q.market_imputed = self.mkt.get_price(crop, state)
        rec.current_price    = mkt_data["current_price"]
        rec.price_volatility = mkt_data["price_volatility"]
        rec.price_trend      = mkt_data["price_trend"]
        # Derive income proxy from live price × land area
        annual_income        = rec.current_price * rec.land_size
        rec.annual_income_proxy = float(annual_income / 100_000)
        rec.market_demand_index  = float(np.clip(0.45 + rec.price_trend * 2, 0, 1))
        rec.export_potential     = float(np.clip(0.25 + rec.price_trend, 0, 1))
        rec.storage_price_premium = float(np.clip(rec.price_volatility * 0.5, 0, 0.4))
        q.source_provenance["market"] = (
            "datagov_agmarknet" if not q.market_imputed else "msp_fallback"
        )

        # ── 5. RBI DBIE ────────────────────────────────────────────────────────
        rbi_data, q.rbi_imputed = self.rbi.get_macro()
        rec.rbi_repo_rate    = rbi_data["rbi_repo_rate"]
        rec.rbi_wpi_inflation = rbi_data["rbi_wpi_inflation"]
        q.source_provenance["macro"] = (
            "rbi_dbie_api" if not q.rbi_imputed else "rbi_hardcoded_fallback"
        )

        # ── 6. Soil health ─────────────────────────────────────────────────────
        soil_params, _ = self.soil.get_from_district(
            str(farmer.get("district", "Unknown"))
        )
        shi, ndr = self.soil.compute_index(
            ph             = soil_params["ph_level"],
            nitrogen       = soil_params["nitrogen"],
            phosphorus     = soil_params["phosphorus"],
            potassium      = soil_params["potassium"],
            organic_carbon = soil_params["organic_carbon"],
        )
        rec.soil_health_index       = shi
        rec.nutrient_deficiency_risk = ndr
        rec.fertilizer_usage_efficiency = shi   # proxy

        # ── 7. Financial ratios from loan history ──────────────────────────────
        fin = self._derive_financial_features(loans, annual_income, rec.land_size)
        for k, v in fin.items():
            setattr(rec, k, v)
        q.loan_imputed = len(loans) == 0
        q.source_provenance["loans"] = "db_loan_records" if loans else "population_default"

        # ── 8. Infrastructure (distance-based) ────────────────────────────────
        # Haversine distance to nearest mandi (using hard-coded major mandi coords)
        mandi_km = self._nearest_mandi_km(lat, lon)
        rec.nearest_mandi_distance = mandi_km
        rec.google_mandi_distance  = mandi_km * 1.12   # road vs crow-flies ~12% overhead
        rec.bank_branch_distance   = mandi_km * 0.70
        rec.connectivity_index     = float(np.clip(1.0 - mandi_km / 40.0 + 0.2, 0.1, 1.0))
        rec.road_quality_index     = float(np.clip(rec.connectivity_index + 0.05, 0.0, 1.0))
        rec.transport_cost_burden  = float(np.clip(mandi_km / 30.0, 0.0, 1.0))

        # ── 9. Derived contextual fields ──────────────────────────────────────
        edu_norm = float(np.clip((rec.education_level - 1) / 4, 0, 1))
        rec.electricity_reliability  = float(np.clip(0.70 + annual_income / 3_000_000, 0, 1))
        rec.mobile_network_strength  = float(np.clip(0.80 + edu_norm * 0.10, 0, 1))
        rec.pest_disease_risk        = float(np.clip(
            rec.drought_risk_7days * 0.3 + (1 - rec.soil_moisture_index) * 0.2
            + (1 - rec.ndvi_current) * 0.15, 0, 1
        ))
        rec.seed_quality_index       = float(np.clip(0.60 + edu_norm * 0.30, 0, 1))
        rec.technology_adoption      = float(np.clip(edu_norm * 0.5 + annual_income / 3_000_000 * 0.3, 0, 1))
        rec.mechanization_level      = float(np.clip(rec.technology_adoption * 0.8, 0, 1))
        rec.insurance_coverage       = float(farmer.get("insurance_coverage", 0.0) or 0.0)
        rec.cooperative_membership   = float(farmer.get("cooperative_membership", 0.0) or 0.0)
        rec.fertilizer_usage_efficiency = rec.soil_health_index
        rec.organic_farming_adoption = 0.15
        rec.precision_agriculture_usage = float(np.clip(rec.technology_adoption * 0.7, 0, 1))
        rec.diversification_index    = float(np.clip(0.25 + rec.land_size / 8 * 0.4, 0.1, 0.9))
        rec.input_cost_index         = float(np.clip(
            0.45 + (1 - rec.connectivity_index) * 0.3
            + rec.price_volatility * 0.2
            + (rec.rbi_wpi_inflation - 4.0) / 20.0, 0.2, 0.9
        ))
        rec.eligible_schemes_count   = float(2 + rec.education_level / 2)
        rec.subsidy_utilization      = float(np.clip(0.3 + 0.4 * rec.cooperative_membership, 0, 1))
        rec.msp_eligibility          = 1.0
        rec.kisan_credit_card        = float(np.clip(0.4 + 0.3 * edu_norm, 0, 1) > 0.5)
        rec.government_training_participation = float(np.clip(0.2 * edu_norm, 0, 1))
        rec.historical_drought_frequency = float(np.clip(
            rec.drought_risk_7days * 4 + 0.5, 0.5, 8.0
        ))
        rec.climate_change_vulnerability = float(np.clip(
            rec.drought_risk_7days * 0.5 + rec.temperature_stress * 0.3
            + (1 - rec.irrigation_access) * 0.2, 0, 1
        ))
        rec.community_leadership_role = 0.0
        rec.social_capital_index     = float(np.clip(
            0.4 + 0.3 * rec.cooperative_membership
            + 0.3 * rec.community_leadership_role, 0, 1
        ))
        rec.extension_service_access = float(np.clip(0.3 + 0.4 * rec.social_capital_index, 0, 1))
        rec.peer_learning_participation = float(np.clip(rec.social_capital_index * 0.7, 0, 1))
        rec.labor_availability       = 0.65
        rec.storage_access           = float(rec.mechanization_level > 0.50)
        rec.supply_chain_integration = float(np.clip(0.2 + 0.5 * rec.cooperative_membership, 0, 1))
        rec.disaster_preparedness    = float(np.clip(
            0.2 + 0.3 * rec.insurance_coverage + 0.3 * edu_norm, 0, 1
        ))
        rec.alternative_income_sources = float(np.clip(0.3 + 0.3 * rec.diversification_index, 0, 1))
        rec.livestock_ownership      = float(rec.land_size > 2.0)

        # Derived features (log-transforms + interaction terms) computed in to_model_df()
        return rec, q

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _derive_financial_features(self, loans: List[dict],
                                   annual_income: float,
                                   land_size: float) -> dict:
        if not loans:
            return dict(
                loan_to_land_ratio=0.20, debt_to_income_ratio=0.30,
                payment_history_score=0.75, savings_to_income_ratio=0.10,
                credit_utilization=0.35, number_of_credit_sources=1.0,
                informal_lending_dependency=0.30,
            )
        loan_amt   = sum(float(l["amount"]) for l in loans
                        if l["status"] in ("active", "defaulted"))
        n_sources  = len(set(l["loan_id"] for l in loans))
        ltr        = loan_amt / (land_size * 150_000 + 1)
        dtr        = loan_amt / (annual_income + 1)
        repaid_ok  = sum(1 for l in loans
                        if l["status"] == "repaid"
                        and float(l["repaid_amount"]) >= float(l["amount"]) * 0.98)
        closed     = sum(1 for l in loans if l["status"] in ("repaid", "defaulted"))
        pay_score  = repaid_ok / closed if closed > 0 else 0.75
        return dict(
            loan_to_land_ratio          = float(np.clip(ltr, 0, 5)),
            debt_to_income_ratio        = float(np.clip(dtr, 0, 2)),
            payment_history_score       = float(np.clip(pay_score, 0.1, 1.0)),
            savings_to_income_ratio     = float(np.clip(0.18 - dtr * 0.15, 0, 0.4)),
            credit_utilization          = float(np.clip(dtr * 1.2, 0, 1)),
            number_of_credit_sources    = float(min(n_sources, 10)),
            informal_lending_dependency = float(np.clip(
                (1 - float(loans[0]["status"] == "repaid")) * 0.4, 0, 0.8
            )),
        )

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        p = math.pi / 180
        a = (math.sin((lat2 - lat1) * p / 2) ** 2
             + math.cos(lat1 * p) * math.cos(lat2 * p)
             * math.sin((lon2 - lon1) * p / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(max(a, 0)))

    # Major APMC mandi lat/lon (representative subset)
    _MANDIS = [
        (28.67, 77.23),   # Delhi Azadpur
        (19.08, 72.88),   # Mumbai Vashi
        (13.08, 77.60),   # Bengaluru Yeshwantpur
        (22.57, 88.36),   # Kolkata
        (17.39, 78.49),   # Hyderabad Bowenpally
        (23.02, 72.57),   # Ahmedabad Jamalpur
        (21.16, 79.11),   # Nagpur
        (26.85, 80.95),   # Lucknow
        (25.59, 85.14),   # Patna
        (30.73, 76.78),   # Chandigarh
    ]

    def _nearest_mandi_km(self, lat: float, lon: float) -> float:
        if not self._MANDIS:
            return 15.0
        dists = [self._haversine(lat, lon, m[0], m[1]) for m in self._MANDIS]
        return float(min(dists))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FEATURE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class FeatureValidator:
    """
    Validates that a DataFrame has exactly the 79 FEATURE_NAMES columns,
    all of dtype float64, with no NaN values.
    Raises ValueError on failure.
    """

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        expected = set(FEATURE_NAMES)
        present  = set(df.columns)
        missing  = expected - present
        extra    = present - expected

        if missing:
            raise ValueError(
                f"FeatureValidator: {len(missing)} required feature(s) missing: "
                f"{sorted(missing)}"
            )
        if extra:
            logger.warning(
                f"FeatureValidator: dropping {len(extra)} extra column(s): {sorted(extra)}"
            )

        result = df[FEATURE_NAMES].astype(np.float64)
        nan_cols = result.columns[result.isnull().any()].tolist()
        if nan_cols:
            raise ValueError(
                f"FeatureValidator: NaN found in feature(s): {nan_cols}"
            )

        assert result.shape[1] == INPUT_DIM, (
            f"FeatureValidator: column count {result.shape[1]} != INPUT_DIM {INPUT_DIM}"
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — HYBRID DATA LOADER  (real + synthetic blend for training)
# ══════════════════════════════════════════════════════════════════════════════

class HybridDataLoader:
    """
    Blends real NABARD-format loan CSV records with synthetic augmentation.

    Real CSV expected columns (minimum):
        farmer_id, loan_amount, status (repaid/active/defaulted),
        disbursed_date, repaid_amount, interest_rate,
        land_size, crop_type, state, education_level,
        family_size, irrigation_access, cooperative_membership,
        insurance_coverage, latitude, longitude
    """

    CROP_ENCODING = {
        "rice": 1, "wheat": 2, "cotton": 3, "sugarcane": 4, "soybean": 5, "maize": 6,
        "Rice": 1, "Wheat": 2, "Cotton": 3, "Sugarcane": 4, "Soybean": 5, "Maize": 6,
    }

    def load(self, real_path: str = REAL_LOAN_DATA_PATH,
             synthetic_gen=None,
             n_synthetic: int = 4000,
             random_seed: int = 42) -> pd.DataFrame:
        """
        Returns a training-ready DataFrame with FEATURE_NAMES + default_flag.
        Output shape: (N, 78).  Column order guaranteed to match FEATURE_NAMES.
        """
        frames  = []
        n_real  = 0

        if real_path and Path(real_path).exists():
            try:
                real_df = pd.read_csv(real_path, low_memory=False)
                real_df = self._map_real_to_schema(real_df, random_seed)
                frames.append(real_df)
                n_real  = len(real_df)
                logger.info(f"Loaded {n_real} real records from {real_path}.")
                logger.info(f"  Real default rate: {real_df['default_flag'].mean():.2%}")
            except Exception as e:
                logger.warning(f"Real data load failed ({e}); falling back to synthetic-only.")

        if synthetic_gen is not None:
            if n_real > 0:
                n_synth = int(n_real * SYNTHETIC_BLEND_RATIO
                              / max(1 - SYNTHETIC_BLEND_RATIO, 0.01))
            else:
                n_synth = n_synthetic
            syn_df = synthetic_gen.generate(n_samples=n_synth, random_seed=random_seed)
            frames.append(syn_df)
            logger.info(f"Added {n_synth} synthetic rows (blend={SYNTHETIC_BLEND_RATIO:.0%}).")

        if not frames:
            raise RuntimeError(
                "HybridDataLoader: no data available.  Set REAL_LOAN_DATA_PATH "
                "or pass a DatasetGenerator via synthetic_gen."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        logger.info(
            f"Combined: {len(combined)} rows, "
            f"default_rate={combined['default_flag'].mean():.2%}, "
            f"real_fraction={n_real / len(combined):.0%}"
        )
        # Final derived features
        combined = self._finalize_derived(combined)
        return combined

    def _map_real_to_schema(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Map real CSV → FEATURE_NAMES + default_flag."""
        rng = np.random.default_rng(seed)
        n   = len(df)
        m   = {}   # mapped columns

        # Label
        if "status" in df.columns:
            m["default_flag"] = (df["status"].str.lower() == "defaulted").astype(int).values
        elif "default_flag" in df.columns:
            m["default_flag"] = df["default_flag"].values
        else:
            raise ValueError("Real CSV must have a 'status' or 'default_flag' column.")

        m["farmer_id"] = (df["farmer_id"].values if "farmer_id" in df.columns
                          else [f"R{i:06d}" for i in range(n)])

        # Direct mappings with fallbacks
        for feat, col, fallback in [
            ("farmer_age",            "farmer_age",            np.clip(rng.gamma(3, 15, n) + 25, 22, 75)),
            ("education_level",       "education_level",       rng.integers(1, 6, n).astype(float)),
            ("family_size",           "family_size",           rng.integers(2, 9, n).astype(float)),
            ("land_size",             "land_size",             rng.gamma(2, 1.5, n)),
            ("irrigation_access",     "irrigation_access",     rng.binomial(1, 0.55, n).astype(float)),
            ("cooperative_membership","cooperative_membership", rng.binomial(1, 0.50, n).astype(float)),
            ("insurance_coverage",    "insurance_coverage",    rng.binomial(1, 0.40, n).astype(float)),
        ]:
            m[feat] = df[col].values.astype(float) if col in df.columns else fallback

        # Crop encoding
        m["crop_type_encoded"] = (
            df["crop_type"].map(self.CROP_ENCODING).fillna(1).values
            if "crop_type" in df.columns
            else rng.choice([1, 2, 3, 4, 5, 6], n).astype(float)
        )

        # Financial ratios from real loan data
        if "loan_amount" in df.columns:
            loan_amt = df["loan_amount"].fillna(0).values.astype(float)
            land     = np.maximum(m["land_size"], 0.1)
            annual_income = np.clip(land * 55_000, 25_000, 3_000_000)
            m["loan_to_land_ratio"]   = loan_amt / (land * 150_000 + 1)
            m["debt_to_income_ratio"] = loan_amt / (annual_income + 1)
            m["annual_income_proxy"]  = annual_income / 100_000
            if "repaid_amount" in df.columns:
                repaid = df["repaid_amount"].fillna(0).values.astype(float)
                m["payment_history_score"] = np.clip(repaid / np.maximum(loan_amt, 1), 0.1, 1.0)
            else:
                m["payment_history_score"] = np.where(
                    m["default_flag"] == 1,
                    rng.uniform(0.10, 0.45, n),
                    rng.uniform(0.55, 1.00, n),
                )
        else:
            m["loan_to_land_ratio"]    = rng.beta(2, 5, n)
            m["debt_to_income_ratio"]  = rng.beta(2, 4, n)
            m["annual_income_proxy"]   = rng.lognormal(0, 0.5, n)
            m["payment_history_score"] = rng.beta(5, 2, n)

        # Synthetic fills for remaining features
        synth_fills = {
            "current_temperature":          rng.normal(30, 4, n),
            "current_humidity":             np.clip(rng.normal(65, 10, n), 30, 95),
            "temperature_stress":           np.clip(rng.beta(2, 8, n) * 0.5, 0, 1),
            "humidity_stress":              np.clip(rng.beta(2, 6, n) * 0.4, 0, 1),
            "drought_risk_7days":           rng.beta(2, 5, n),
            "frost_risk_7days":             rng.beta(1, 12, n),
            "excess_rain_risk":             rng.beta(1, 9, n),
            "soil_moisture_index":          rng.beta(4, 4, n),
            "ndvi_current":                 rng.beta(4, 3, n),
            "ndvi_anomaly":                 rng.normal(0, 0.15, n),
            "price_volatility":             np.clip(rng.normal(0.20, 0.08, n), 0.05, 0.50),
            "current_price":                rng.lognormal(9.5, 0.5, n),
            "market_demand_index":          rng.beta(4, 5, n),
            "export_potential":             rng.beta(2, 6, n),
            "storage_price_premium":        rng.beta(2, 7, n),
            "price_trend":                  rng.normal(0.02, 0.10, n),
            "rbi_repo_rate":                np.full(n, 6.50),
            "rbi_wpi_inflation":            np.full(n, 4.50),
            "savings_to_income_ratio":      np.clip(rng.beta(3, 8, n) * 0.4, 0, 0.4),
            "credit_utilization":           rng.beta(3, 5, n),
            "number_of_credit_sources":     rng.poisson(1.5, n).astype(float),
            "informal_lending_dependency":  rng.beta(2, 6, n),
            "log_debt_to_income":           np.zeros(n),
            "log_loan_to_land":             np.zeros(n),
            "log_annual_income":            np.zeros(n),
            "log_mandi_distance":           np.zeros(n),
            "yield_consistency":            rng.beta(6, 4, n),
            "soil_health_index":            rng.beta(5, 4, n),
            "nutrient_deficiency_risk":     rng.beta(3, 6, n),
            "nearest_mandi_distance":       rng.gamma(2.5, 8, n),
            "connectivity_index":           rng.beta(5, 5, n),
            "road_quality_index":           rng.beta(5, 5, n),
            "electricity_reliability":      rng.beta(6, 4, n),
            "mobile_network_strength":      rng.beta(8, 3, n),
            "bank_branch_distance":         rng.gamma(2, 5, n),
            "transport_cost_burden":        rng.beta(3, 6, n),
            "google_mandi_distance":        rng.gamma(2.5, 8, n),
            "technology_adoption":          rng.beta(4, 6, n),
            "diversification_index":        rng.beta(3, 6, n),
            "input_cost_index":             rng.beta(4, 5, n),
            "mechanization_level":          rng.beta(3, 6, n),
            "seed_quality_index":           rng.beta(5, 4, n),
            "fertilizer_usage_efficiency":  rng.beta(5, 4, n),
            "pest_disease_risk":            rng.beta(2, 7, n),
            "organic_farming_adoption":     rng.beta(2, 8, n),
            "precision_agriculture_usage":  rng.beta(2, 8, n),
            "eligible_schemes_count":       rng.poisson(2, n).astype(float),
            "subsidy_utilization":          rng.beta(3, 7, n),
            "msp_eligibility":              rng.binomial(1, 0.70, n).astype(float),
            "kisan_credit_card":            rng.binomial(1, 0.40, n).astype(float),
            "government_training_participation": rng.beta(2, 8, n),
            "seasonal_rainfall_deviation":  rng.normal(0, 18, n),
            "historical_drought_frequency": rng.poisson(1.5, n).astype(float),
            "climate_change_vulnerability": rng.beta(3, 6, n),
            "community_leadership_role":    rng.binomial(1, 0.10, n).astype(float),
            "social_capital_index":         rng.beta(4, 5, n),
            "extension_service_access":     rng.beta(3, 6, n),
            "peer_learning_participation":  rng.beta(3, 6, n),
            "labor_availability":           rng.beta(5, 4, n),
            "storage_access":               rng.binomial(1, 0.30, n).astype(float),
            "supply_chain_integration":     rng.beta(3, 6, n),
            "disaster_preparedness":        rng.beta(3, 6, n),
            "alternative_income_sources":   rng.beta(3, 6, n),
            "livestock_ownership":          rng.binomial(1, 0.50, n).astype(float),
            "seasonal_payment_consistency": np.zeros(n),
            "repayment_velocity_proxy":     np.zeros(n),
            "climate_debt_compound_stress": np.zeros(n),
        }
        for feat, arr in synth_fills.items():
            if feat not in m:
                m[feat] = arr

        out = pd.DataFrame({"farmer_id": m["farmer_id"]})
        for feat in FEATURE_NAMES:
            out[feat] = m[feat]
        out["default_flag"] = m["default_flag"]
        return out

    @staticmethod
    def _finalize_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Overwrite derived columns with freshly computed values."""
        df = df.copy()
        df["log_debt_to_income"] = np.log1p(df["debt_to_income_ratio"])
        df["log_loan_to_land"]   = np.log1p(df["loan_to_land_ratio"])
        df["log_annual_income"]  = np.log1p(df["annual_income_proxy"])
        df["log_mandi_distance"] = np.log1p(df["nearest_mandi_distance"])

        df["seasonal_payment_consistency"] = np.clip(
            df["payment_history_score"] * 0.7
            + (1.0 - df["price_volatility"]) * 0.3, 0, 1
        )
        dti_safe = np.maximum(df["debt_to_income_ratio"].values, 0.05)
        df["repayment_velocity_proxy"] = np.clip(
            df["payment_history_score"].values / dti_safe * 0.7, 0, 1
        )
        # NDVI-augmented compound stress (novel contribution)
        ndvi_w = np.clip(1.0 - df["ndvi_current"].values, 0.2, 1.0)
        df["climate_debt_compound_stress"] = np.clip(
            np.sqrt(
                np.maximum(df["drought_risk_7days"].values, 0)
                * np.maximum(df["debt_to_income_ratio"].values, 0)
                * ndvi_w
                * (1.0 + df["rbi_wpi_inflation"].values / 20.0)
            ), 0, 1
        )
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class AgroScoreInferencePipeline:
    """Wraps a trained AgroScoreModel for per-farmer real-time scoring."""

    def __init__(self, model, db: Optional[AgroScoreDatabase],
                 builder: Optional[AgroScorePipelineFeatureBuilder]):
        self.model   = model
        self.db      = db
        self.builder = builder

    @classmethod
    def from_trained_model(cls, model,
                           db_path: str = DATABASE_PATH) -> "AgroScoreInferencePipeline":
        db      = AgroScoreDatabase(db_path)
        ow      = OpenWeatherClient(db=db)
        tmr     = TomorrowIOClient(db=db)
        nasa    = NASAEarthdataClient(db=db)
        mkt     = DataGovMarketClient(db=db)
        rbi_cli = RBIClient(db=db)
        soil    = SoilHealthComputer()
        builder = AgroScorePipelineFeatureBuilder(db, ow, tmr, nasa, mkt, rbi_cli, soil)
        return cls(model, db, builder)

    def score(self, farmer_id: int) -> dict:
        rec, quality = self.builder.build(farmer_id)
        df           = FeatureValidator.validate(rec.to_model_df())
        result       = self.model.predict(df)
        result["data_quality"] = {
            "confidence":          quality.overall_confidence,
            "openweather_imputed": quality.openweather_imputed,
            "tomorrow_imputed":    quality.tomorrow_imputed,
            "ndvi_imputed":        quality.ndvi_imputed,
            "market_imputed":      quality.market_imputed,
            "rbi_imputed":         quality.rbi_imputed,
            "loan_imputed":        quality.loan_imputed,
            "source_provenance":   quality.source_provenance,
            "notes":               quality.imputation_notes,
        }
        return result

    def score_from_dict(self, feature_dict: dict) -> dict:
        """Score directly from a dict of feature values (bypasses API calls)."""
        valid_fields = set(FarmerFeatureRecord.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in feature_dict.items() if k in valid_fields}
        rec    = FarmerFeatureRecord(**kwargs)
        df     = FeatureValidator.validate(rec.to_model_df())
        result = self.model.predict(df)
        result["data_quality"] = {"confidence": "HIGH", "notes": ["Direct dict input."]}
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("AgroScore Pipeline v3.0 — refactored production edition")
    print(f"  Feature count  : {INPUT_DIM}  (INPUT_DIM exported for model.py)")
    print(f"  Real API sources: {list(REAL_DATA_FEATURES.keys())}")
    print(f"  Cache enabled   : {CACHE_ENABLED}  (TTL {CACHE_TTL}s)")
    print(f"  Database path   : {DATABASE_PATH}")
    # Quick smoke test — build a default record and validate it
    rec = FarmerFeatureRecord()
    df  = FeatureValidator.validate(rec.to_model_df())
    print(f"  Smoke test      : FarmerFeatureRecord → shape {df.shape} ✓")
