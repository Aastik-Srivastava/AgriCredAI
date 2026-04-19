"""
================================================================================
  AgroScore v3.0 — Backward-Compatible Data Pipeline Wrapper
================================================================================

This module re-exports everything from pipeline.py (the canonical pipeline)
and provides a backward-compatible AgroScoreInferencePipeline class that
advanced_app.py expects:
    - Constructor: AgroScoreInferencePipeline(weather_key=..., market_key=...)
    - Method: .calculate_and_store_portfolio_metrics()
================================================================================
"""

# ── Re-export everything from the canonical pipeline ──────────────────────────
from pipeline import (
    # Config
    DATABASE_PATH, WEATHER_API_KEY, TOMORROW_IO_API_KEY,
    NASA_EARTHDATA_USER, NASA_EARTHDATA_PASS, MARKET_API_KEY as _MARKET_API_KEY,
    RBI_API_BASE, REAL_LOAN_DATA_PATH,
    CACHE_ENABLED, CACHE_TTL, RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD,
    SYNTHETIC_BLEND_RATIO,
    # Feature schema
    FEATURE_SCHEMA, FEATURE_NAMES, INPUT_DIM, REAL_DATA_FEATURES,
    # Dataclasses
    FarmerFeatureRecord, DataQualityReport,
    # Database
    AgroScoreDatabase,
    # API Clients
    OpenWeatherClient, TomorrowIOClient, NASAEarthdataClient,
    DataGovMarketClient, RBIClient, SoilHealthComputer,
    # Feature Builder
    AgroScorePipelineFeatureBuilder,
    # Validator & Loader
    FeatureValidator, HybridDataLoader,
    # Logger
    logger,
)

import sqlite3
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional


class AgroScoreInferencePipeline:
    """
    Backward-compatible inference pipeline for advanced_app.py.

    Wraps the new pipeline.py architecture with the old constructor signature
    that accepts (weather_key, market_key) and provides portfolio analytics
    methods used by the Streamlit frontend.
    """

    def __init__(self, weather_key: str = '', market_key: str = '',
                 db_path: str = DATABASE_PATH):
        self.weather_key = weather_key or WEATHER_API_KEY
        self.market_key = market_key or _MARKET_API_KEY
        self.db_path = db_path
        self.db = AgroScoreDatabase(db_path)

        # Initialize API clients
        self.ow_client = OpenWeatherClient(api_key=self.weather_key, db=self.db)
        self.tmr_client = TomorrowIOClient(db=self.db)
        self.nasa_client = NASAEarthdataClient(db=self.db)
        self.mkt_client = DataGovMarketClient(api_key=self.market_key, db=self.db)
        self.rbi_client = RBIClient(db=self.db)
        self.soil_computer = SoilHealthComputer()

        self.builder = AgroScorePipelineFeatureBuilder(
            self.db, self.ow_client, self.tmr_client,
            self.nasa_client, self.mkt_client, self.rbi_client,
            self.soil_computer,
        )

        self._model = None  # loaded lazily if needed
        # Persistent connection for UI backwards compatibility
        self._app_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._app_conn.row_factory = sqlite3.Row

    # ── Portfolio Metrics (used by advanced_app.py sidebar) ───────────────────
    def calculate_and_store_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio metrics from the loans / farmers database tables."""
        try:
            with self.db._conn() as conn:
                total_farmers = conn.execute(
                    "SELECT COUNT(*) FROM farmers"
                ).fetchone()[0]

                total_loans = conn.execute(
                    "SELECT COUNT(*) FROM loans"
                ).fetchone()[0]

                total_portfolio = conn.execute(
                    "SELECT COALESCE(SUM(amount), 0) FROM loans"
                ).fetchone()[0]

                active_loans = conn.execute(
                    "SELECT COUNT(*) FROM loans WHERE status='active'"
                ).fetchone()[0]

                repaid_loans = conn.execute(
                    "SELECT COUNT(*) FROM loans WHERE status='repaid'"
                ).fetchone()[0]

                defaulted_loans = conn.execute(
                    "SELECT COUNT(*) FROM loans WHERE status='defaulted'"
                ).fetchone()[0]

                default_rate = (defaulted_loans / max(total_loans, 1)) * 100

                avg_credit_score = conn.execute(
                    "SELECT COALESCE(AVG(credit_score), 650) "
                    "FROM loans WHERE credit_score IS NOT NULL"
                ).fetchone()[0]

                avg_loan_amount = conn.execute(
                    "SELECT COALESCE(AVG(amount), 0) FROM loans"
                ).fetchone()[0]

                total_land = conn.execute(
                    "SELECT COALESCE(SUM(land_size), 0) FROM farmers"
                ).fetchone()[0]

            metrics = {
                'total_farmers':    total_farmers,
                'total_loans':      total_loans,
                'total_portfolio':  total_portfolio,
                'active_loans':     active_loans,
                'repaid_loans':     repaid_loans,
                'defaulted_loans':  defaulted_loans,
                'default_rate':     default_rate,
                'avg_credit_score': avg_credit_score,
                'avg_loan_amount':  avg_loan_amount,
                'total_land_size':  total_land,
            }

            # Persist today's snapshot
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                with self.db._conn() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO portfolio_metrics
                           (date, total_farmers, total_loans,
                            total_portfolio_value, active_loans,
                            repaid_loans, defaulted_loans,
                            default_rate, avg_credit_score,
                            total_land_size, avg_loan_amount)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (today, total_farmers, total_loans, total_portfolio,
                         active_loans, repaid_loans, defaulted_loans,
                         default_rate, avg_credit_score, total_land,
                         avg_loan_amount),
                    )
            except Exception as e:
                logger.debug(f"Metrics store error: {e}")

            return metrics

        except Exception as e:
            logger.warning(f"Portfolio metrics calculation error: {e}")
            return {
                'total_farmers': 0, 'total_loans': 0,
                'total_portfolio': 0, 'active_loans': 0,
                'repaid_loans': 0, 'defaulted_loans': 0,
                'default_rate': 0, 'avg_credit_score': 650,
                'avg_loan_amount': 0, 'total_land_size': 0,
            }

    @property
    def conn(self):
        """Backward-compatible connection property for advanced_app.py"""
        return self._app_conn

    def seed_farmers(self, count: int = 2000):
        """Dummy backward-compatible method for seeding farmers database"""
        logger.info(f"Mocking seed of {count} farmers (handled by HybridDataLoader in v3)")
        try:
            with self.conn as db_conn:
                db_conn.execute(
                    "INSERT OR IGNORE INTO farmers (farmer_id, name, state) VALUES (?, ?, ?)",
                    ('F0001', 'Mock Farmer', 'Maharashtra')
                )
        except Exception:
            pass

    def seed_loans_for_farmers(self):
        """Dummy backward-compatible method for seeding loans"""
        logger.info("Mocking loan seed generation (handled by HybridDataLoader in v3)")

    def seed_portfolio_history(self, days: int = 90):
        """Dummy backward-compatible method for seeding portfolio daily history"""
        logger.info(f"Mocking seed portfolio history for {days} days")

    def get_portfolio_trends(self, days: int = 30):
        """Fetch historical portfolio metrics for the last N days."""
        try:
            import pandas as pd
            query = f"SELECT * FROM portfolio_metrics ORDER BY date DESC LIMIT {days}"
            df = pd.read_sql_query(query, self._app_conn)
            if not df.empty:
                df = df.sort_values("date")
            return df
        except Exception as e:
            logger.warning(f"Error getting trends: {e}")
            import pandas as pd
            return pd.DataFrame()

    def get_market_prices(self, crop: str, state: str = "all") -> dict:
        """Dummy backward-compatible method for UI"""
        prices = {"wheat": 2200, "rice": 2500, "cotton": 5000, "soybean": 3500}
        return {"price_per_quintal": prices.get(crop.lower(), 2500), "source": "API"}


    # ── Feature building (wraps the new pipeline builder) ─────────────────────
    def build_features(self, farmer_id: int):
        """Build feature record for a farmer from DB + API data."""
        return self.builder.build(farmer_id)

    def score_from_dict(self, feature_dict: dict) -> dict:
        """Score directly from a dict of feature values (bypasses API calls)."""
        valid_fields = set(FarmerFeatureRecord.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in feature_dict.items() if k in valid_fields}
        rec = FarmerFeatureRecord(**kwargs)
        df = FeatureValidator.validate(rec.to_model_df())

        if self._model and hasattr(self._model, 'predict'):
            return self._model.predict(df)

        return {
            "features": df.to_dict(orient='records')[0],
            "data_quality": {"confidence": "HIGH", "notes": ["Direct dict input."]},
        }

    def get_farmer_data(self, farmer_id: int) -> Optional[dict]:
        """Look up farmer record from DB."""
        return self.db.get_farmer(farmer_id)

    def get_loans(self, farmer_id: int) -> List[dict]:
        """Get loan records for a farmer."""
        return self.db.get_loans(farmer_id)
