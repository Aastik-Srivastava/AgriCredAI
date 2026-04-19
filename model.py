"""
================================================================================
  AgroScore v3.0 — Agricultural Credit Scoring Model  (REFACTORED)
  Production-ready, pipeline-integrated edition
================================================================================

Pipeline contract
-----------------
  Imports INPUT_DIM (= 77) and FEATURE_NAMES from pipeline.py.
  All internal objects that consume features reference INPUT_DIM or FEATURE_NAMES
  directly, so a pipeline schema change automatically propagates here.

Model architecture
------------------
  Stacking ensemble (StackingClassifier):
    Base learners:
      • XGBoost  (xgb.XGBClassifier)
      • LightGBM (lgb.LGBMClassifier)
      • Random Forest (sklearn)
    Meta-learner:
      • Calibrated Logistic Regression (sigmoid calibration, 3-fold CV)

  Post-processing:
    • PDO score transform  (300-point scale, doubled odds at 50 points)
    • Policy Adjustment Layer (rule-based overrides for regulatory compliance)

Research capabilities (run via flags in train()):
  • BenchmarkSuite  — 5-model comparison table (LaTeX-ready)
  • AblationStudy   — 6 ablation experiments (Δ AUC / KS / Gini)
  • FullFairnessAudit — 5 fairness metrics with reweighting mitigation
  • SHAP bootstrap stability analysis
  • Decision consistency check (noise robustness)
================================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from scipy.stats import wasserstein_distance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
    precision_score, recall_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut, StratifiedKFold, TimeSeriesSplit,
    cross_val_predict, cross_val_score,
)
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Pipeline integration — single source of truth ────────────────────────────
from pipeline import (
    INPUT_DIM,          # int = 79
    FEATURE_NAMES,      # List[str], len 79, ordered
    FEATURE_SCHEMA,     # Dict[str, dtype]
    FeatureValidator,
    FarmerFeatureRecord,
    HybridDataLoader,
)

# Compile-time guard: fail fast if schema drifts
assert INPUT_DIM == 79, f"pipeline.INPUT_DIM={INPUT_DIM}, expected 79"
assert len(FEATURE_NAMES) == INPUT_DIM, (
    f"len(FEATURE_NAMES)={len(FEATURE_NAMES)} != INPUT_DIM={INPUT_DIM}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SCHEMA ALIGNMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_schema_alignment(verbose: bool = True) -> bool:
    """Verify that DatasetGenerator output columns match FEATURE_NAMES."""
    gen    = DatasetGenerator()
    sample = gen.generate(n_samples=10, random_seed=0)
    gen_cols    = set(sample.columns) - {"farmer_id", "default_flag"}
    schema_cols = set(FEATURE_NAMES)
    only_schema = schema_cols - gen_cols
    only_gen    = gen_cols - schema_cols
    aligned     = (not only_schema) and (not only_gen)
    if verbose:
        if aligned:
            print(f"  [schema_check] ALIGNED — {INPUT_DIM} features (INPUT_DIM={INPUT_DIM}).")
        else:
            if only_schema:
                print(f"  [schema_check] In schema but not generator: {sorted(only_schema)}")
            if only_gen:
                print(f"  [schema_check] In generator but not schema: {sorted(only_gen)}")
    return aligned


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATASET GENERATOR  (synthetic baseline / ablation control)
# ══════════════════════════════════════════════════════════════════════════════

class DatasetGenerator:
    """
    Generates synthetic agricultural training data strictly aligned with
    FEATURE_NAMES (imported from pipeline.py).  Column count guaranteed
    to equal INPUT_DIM after generate() is called.

    Retained for ablation studies and researchers without real loan access.
    Production training uses HybridDataLoader (real + synthetic blend).
    """

    REGION_MAPPING = {
        "Punjab":      {"avg_land": 4.2, "irrigation_p": 0.95, "income_mult": 1.30},
        "Maharashtra": {"avg_land": 2.8, "irrigation_p": 0.45, "income_mult": 1.10},
        "UP":          {"avg_land": 1.9, "irrigation_p": 0.75, "income_mult": 0.90},
        "Karnataka":   {"avg_land": 3.1, "irrigation_p": 0.35, "income_mult": 1.00},
        "AP":          {"avg_land": 2.4, "irrigation_p": 0.55, "income_mult": 1.05},
        "WB":          {"avg_land": 1.6, "irrigation_p": 0.85, "income_mult": 0.85},
        "Gujarat":     {"avg_land": 3.5, "irrigation_p": 0.65, "income_mult": 1.20},
        "MP":          {"avg_land": 4.8, "irrigation_p": 0.30, "income_mult": 0.95},
    }
    CROP_PROBS = {
        "Punjab":      [0.10, 0.60, 0.05, 0.15, 0.05, 0.05],
        "Maharashtra": [0.15, 0.20, 0.30, 0.20, 0.10, 0.05],
        "UP":          [0.30, 0.40, 0.10, 0.10, 0.05, 0.05],
        "Karnataka":   [0.20, 0.15, 0.20, 0.25, 0.15, 0.05],
        "AP":          [0.25, 0.10, 0.25, 0.25, 0.10, 0.05],
        "WB":          [0.70, 0.15, 0.05, 0.05, 0.03, 0.02],
        "Gujarat":     [0.10, 0.20, 0.40, 0.15, 0.10, 0.05],
        "MP":          [0.15, 0.35, 0.15, 0.15, 0.15, 0.05],
    }
    DROUGHT_BASE  = {"Punjab": 0.20, "Maharashtra": 0.60, "UP": 0.30,
                     "Karnataka": 0.70, "AP": 0.50, "WB": 0.20,
                     "Gujarat": 0.40, "MP": 0.80}
    COOP_BASE     = {"Punjab": 0.75, "Maharashtra": 0.55, "UP": 0.40,
                     "Karnataka": 0.50, "AP": 0.45, "WB": 0.35,
                     "Gujarat": 0.80, "MP": 0.40}
    PRICE_VOL     = {1: 0.18, 2: 0.12, 3: 0.35, 4: 0.25, 5: 0.28, 6: 0.20}
    CROP_INC_MULT = {1: 1.00, 2: 0.85, 3: 1.40, 4: 1.60, 5: 1.15, 6: 0.95}
    REGIONAL_TEMP = {"Punjab": 28, "Maharashtra": 32, "UP": 30, "Karnataka": 29,
                     "AP": 33, "WB": 31, "Gujarat": 34, "MP": 31}
    REGIONAL_HUM  = {"Punjab": 55, "Maharashtra": 65, "UP": 70, "Karnataka": 62,
                     "AP": 68, "WB": 78, "Gujarat": 58, "MP": 60}

    def generate(self, n_samples: int = 8000, random_seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed)
        n   = n_samples
        d   = {}

        regions    = rng.choice(list(self.REGION_MAPPING), size=n)
        farmer_ids = [f"F{i:06d}" for i in range(n)]

        # ── Demographics ──────────────────────────────────────────────────────
        ages                   = rng.gamma(3, 15, n) + 25
        d["farmer_age"]        = np.clip(ages, 22, 75)
        age_edu                = np.where(d["farmer_age"] < 35, 0.4,
                                  np.where(d["farmer_age"] < 50, 0.0, -0.3))
        d["education_level"]   = np.clip((rng.beta(2, 3, n) + age_edu) * 5, 1, 5).astype(float)
        d["family_size"]       = rng.poisson(
            np.clip(4.2 + 0.05 * (d["farmer_age"] - 45)
                    - 0.3 * (d["education_level"] - 2.5), 2, 8)
        ).astype(float)

        # ── Land & crop ───────────────────────────────────────────────────────
        rl                     = np.array([self.REGION_MAPPING[r]["avg_land"] for r in regions])
        d["land_size"]         = np.clip(rl * rng.gamma(2, 0.5, n), 0.5, 20)
        d["crop_type_encoded"] = np.array(
            [rng.choice(range(1, 7), p=self.CROP_PROBS[r]) for r in regions], dtype=float
        )
        irr_p                  = np.clip(
            np.array([self.REGION_MAPPING[r]["irrigation_p"] for r in regions])
            + rng.normal(0, 0.1, n), 0.1, 0.95
        )
        d["irrigation_access"] = rng.binomial(1, irr_p, n).astype(float)

        # ── Weather (real API proxy for training) ──────────────────────────────
        bt = np.array([self.REGIONAL_TEMP[r] for r in regions])
        d["current_temperature"] = bt + rng.normal(0, 4, n)
        bh = np.array([self.REGIONAL_HUM[r] for r in regions])
        d["current_humidity"]  = np.clip(bh + rng.normal(0, 10, n), 30, 95)
        d["temperature_stress"]= np.clip((d["current_temperature"] - 28) / 15, 0, 1) \
                                  + rng.beta(2, 8, n) * 0.3
        d["humidity_stress"]   = np.clip(np.abs(d["current_humidity"] - 60) / 25, 0, 1) \
                                  + rng.beta(2, 6, n) * 0.2
        drought_base           = np.array([self.DROUGHT_BASE[r] for r in regions])
        d["drought_risk_7days"]= np.clip(
            drought_base * rng.beta(2, 5, n) - d["irrigation_access"] * 0.3, 0, 1
        )
        fs = np.isin(d["crop_type_encoded"], [1, 3, 4, 5])
        nb = np.isin(regions, ["Punjab", "UP", "MP"])
        fb = np.where(fs, 0.4, 0.05); fb = np.where(nb, fb * 2, fb)
        d["frost_risk_7days"]  = np.clip(fb * rng.beta(1, 12, n), 0, 1)
        d["excess_rain_risk"]  = rng.beta(1, 9, n) * (1 - d["drought_risk_7days"] * 0.8)

        # ── New v3 real-API proxy features ────────────────────────────────────
        d["soil_moisture_index"] = np.clip(
            rng.beta(4, 4, n) * (1 - d["drought_risk_7days"] * 0.5), 0, 1
        )
        d["ndvi_current"]        = np.clip(
            0.3 + d["irrigation_access"] * 0.3
            + (1 - d["drought_risk_7days"]) * 0.25
            + rng.beta(3, 4, n) * 0.15, 0.1, 0.9
        )
        d["ndvi_anomaly"]        = rng.normal(0, 0.15, n)

        # ── Market ────────────────────────────────────────────────────────────
        vb = np.array([self.PRICE_VOL[int(c)] for c in d["crop_type_encoded"]])
        d["price_volatility"]   = np.clip(vb + rng.beta(2, 6, n) * 0.3 - 0.15, 0.05, 0.80)
        ri = np.array([self.REGION_MAPPING[r]["income_mult"] for r in regions])
        cm = np.array([self.CROP_INC_MULT[int(c)] for c in d["crop_type_encoded"]])
        annual_income = np.clip(
            d["land_size"] * 55_000 * cm * ri
            * (1 + 0.5 * d["irrigation_access"])
            * rng.lognormal(0, 0.4, n),
            25_000, 3_000_000
        )
        d["annual_income_proxy"] = annual_income / 100_000
        d["current_price"]       = annual_income / d["land_size"] + rng.normal(0, 8000, n)
        d["market_demand_index"] = rng.beta(4, 5, n)
        d["export_potential"]    = np.where(
            np.isin(d["crop_type_encoded"], [2, 3, 4]),
            rng.beta(5, 5, n), rng.beta(2, 7, n)
        )
        d["storage_price_premium"] = rng.beta(2, 7, n)
        d["price_trend"]           = rng.normal(0.02, 0.12, n)

        # ── RBI macro ─────────────────────────────────────────────────────────
        d["rbi_repo_rate"]     = np.clip(rng.normal(6.5, 0.8, n),  4.0,  9.0)
        d["rbi_wpi_inflation"] = np.clip(rng.normal(4.5, 2.0, n),  0.5, 15.0)

        # ── Financial ratios ──────────────────────────────────────────────────
        lp        = np.clip(0.75 + 0.15 * (d["land_size"] / 10)
                            + 0.10 * (d["education_level"] / 5), 0.3, 0.95)
        has_loan  = rng.binomial(1, lp, n)
        loan_amt  = np.where(has_loan, annual_income * rng.beta(3, 4, n) * 0.8, 0)
        d["loan_to_land_ratio"]    = loan_amt / (d["land_size"] * 150_000 + 1)
        d["debt_to_income_ratio"]  = loan_amt / (annual_income + 1)
        fin_stress = np.clip(d["debt_to_income_ratio"] + d["loan_to_land_ratio"] / 2, 0, 2)
        inc_stab   = 1 - d["price_volatility"] - d["drought_risk_7days"] * 0.5
        d["payment_history_score"] = np.clip(
            0.88 - fin_stress * 0.25 + inc_stab * 0.15
            + d["education_level"] / 10 + rng.normal(0, 0.12, n), 0.1, 1.0
        )
        d["savings_to_income_ratio"] = np.clip(
            0.18 - d["debt_to_income_ratio"] * 0.15
            + (d["education_level"] / 5) * 0.08 + rng.beta(3, 8, n) * 0.2, 0, 0.4
        )
        d["credit_utilization"]      = np.clip(
            d["debt_to_income_ratio"] * 1.2 + rng.beta(3, 6, n) * 0.3, 0, 1
        )
        d["number_of_credit_sources"] = rng.poisson(
            1.2 + d["education_level"] / 5 + has_loan * 0.5, n
        ).astype(float)

        # ── Log-transforms (computed later in _finalize) ──────────────────────
        d["log_debt_to_income"] = np.zeros(n)
        d["log_loan_to_land"]   = np.zeros(n)
        d["log_annual_income"]  = np.zeros(n)
        d["nearest_mandi_distance"] = rng.gamma(2.5, 8, n)
        d["log_mandi_distance"] = np.zeros(n)

        # ── Yield & soil ──────────────────────────────────────────────────────
        ws = 1 - (d["drought_risk_7days"] + d["frost_risk_7days"] + d["excess_rain_risk"]) / 3
        d["yield_consistency"]       = np.clip(
            0.7 + d["irrigation_access"] * 0.3 + ws * 0.2
            + d["education_level"] / 20 + rng.beta(6, 4, n) * 0.2 - 0.1, 0.3, 1.0
        )
        d["soil_health_index"]       = np.clip(
            0.65 + d["irrigation_access"] * 0.2
            + (d["education_level"] / 5) * 0.1
            + rng.beta(4, 3, n) * 0.25 - 0.1, 0.2, 1.0
        )
        d["nutrient_deficiency_risk"]= np.clip(
            1.2 - d["soil_health_index"] - (annual_income / 500_000) * 0.3
            + rng.beta(3, 5, n) * 0.4, 0, 1
        )

        # ── Infrastructure ────────────────────────────────────────────────────
        d["connectivity_index"]     = np.clip(
            1 - d["nearest_mandi_distance"] / 40 + rng.beta(5, 5, n) * 0.4, 0.1, 1.0
        )
        d["road_quality_index"]     = np.clip(d["connectivity_index"] + rng.beta(3, 7, n) * 0.3, 0, 1)
        d["electricity_reliability"]= np.clip(
            0.7 + 0.2 * annual_income / 300_000 + rng.beta(4, 6, n) * 0.3, 0, 1
        )
        d["mobile_network_strength"]= np.clip(0.8 + rng.beta(5, 5, n) * 0.2, 0, 1)
        d["bank_branch_distance"]   = d["nearest_mandi_distance"] * 0.7 + rng.gamma(2, 3, n)
        d["transport_cost_burden"]  = np.clip(
            d["nearest_mandi_distance"] / 30 + rng.beta(3, 7, n) * 0.5, 0, 1
        )
        d["google_mandi_distance"]  = d["nearest_mandi_distance"] * (1 + rng.normal(0, 0.15, n))

        # ── Support & social ──────────────────────────────────────────────────
        ins_p = np.clip(
            0.35 + (d["education_level"] / 5) * 0.3 + (annual_income / 500_000) * 0.2,
            0.15, 0.85
        )
        d["insurance_coverage"]    = rng.binomial(1, ins_p, n).astype(float)
        coop_p = np.clip(
            np.array([self.COOP_BASE[r] for r in regions])
            + (d["education_level"] / 5 - 0.6) * 0.2, 0.05, 0.95
        )
        d["cooperative_membership"]= rng.binomial(1, coop_p, n).astype(float)
        tech = (d["education_level"] / 5) * 0.4 \
               + (annual_income / 300_000) * 0.3 \
               + d["irrigation_access"] * 0.2
        d["technology_adoption"]   = np.clip(tech + rng.beta(3, 6, n) * 0.4, 0.1, 0.95)
        d["diversification_index"] = np.clip(
            0.25 + (d["land_size"] / 8) * 0.4
            + (d["education_level"] / 5) * 0.2 + rng.beta(3, 6, n) * 0.35, 0.1, 0.9
        )
        d["input_cost_index"]      = np.clip(
            0.45 + (1 - d["connectivity_index"]) * 0.3
            + d["price_volatility"] * 0.2 + rng.beta(4, 5, n) * 0.25, 0.2, 0.9
        )
        d["mechanization_level"]   = np.clip(
            d["technology_adoption"] * 0.8 + (d["land_size"] / 10) * 0.2, 0, 1
        )
        d["seed_quality_index"]    = np.clip(
            0.6 + 0.3 * d["technology_adoption"] + rng.beta(4, 6, n) * 0.3, 0, 1
        )
        d["fertilizer_usage_efficiency"] = np.clip(
            d["soil_health_index"] + rng.beta(4, 6, n) * 0.3, 0, 1
        )
        d["pest_disease_risk"]     = np.clip(
            rng.beta(2, 7, n) * (1 + d["drought_risk_7days"] * 0.3), 0, 1
        )
        d["organic_farming_adoption"]    = rng.beta(2, 8, n)
        d["precision_agriculture_usage"] = np.clip(
            d["technology_adoption"] * 0.7 + rng.beta(1, 9, n) * 0.3, 0, 1
        )
        d["informal_lending_dependency"] = np.clip(
            (1 - d["cooperative_membership"]) * 0.4
            + (1 - d["insurance_coverage"]) * 0.2
            + rng.beta(2, 6, n) * 0.4, 0, 0.8
        )

        # ── Government / scheme ───────────────────────────────────────────────
        d["eligible_schemes_count"]    = rng.poisson(
            2 + d["education_level"] / 2, n
        ).astype(float)
        d["subsidy_utilization"]       = np.clip(
            0.3 + 0.4 * d["cooperative_membership"] + rng.beta(3, 7, n) * 0.4, 0, 1
        )
        d["msp_eligibility"]           = rng.binomial(1, 0.7, n).astype(float)
        d["kisan_credit_card"]         = rng.binomial(
            1, 0.4 + 0.3 * (d["education_level"] / 5), n
        ).astype(float)
        d["government_training_participation"] = np.clip(
            0.2 * d["education_level"] / 5 + rng.beta(2, 8, n), 0, 1
        )

        # ── Climate / seasonal ────────────────────────────────────────────────
        d["seasonal_rainfall_deviation"] = rng.normal(0, 18, n)
        d["historical_drought_frequency"]= rng.poisson(
            drought_base * 4 + 0.8, n
        ).astype(float)
        d["climate_change_vulnerability"]= (
            d["drought_risk_7days"] * 0.5
            + d["temperature_stress"] * 0.3
            + (1 - d["irrigation_access"]) * 0.2
        )

        # ── Community ─────────────────────────────────────────────────────────
        d["community_leadership_role"]  = rng.binomial(
            1, 0.1 + 0.1 * d["education_level"] / 5, n
        ).astype(float)
        d["social_capital_index"]       = np.clip(
            0.4 + 0.3 * d["cooperative_membership"]
            + 0.3 * d["community_leadership_role"] + rng.beta(4, 6, n) * 0.3, 0, 1
        )
        d["extension_service_access"]   = np.clip(
            0.3 + 0.4 * d["social_capital_index"] + rng.beta(3, 7, n) * 0.3, 0, 1
        )
        d["peer_learning_participation"]= np.clip(
            d["social_capital_index"] * 0.7 + rng.beta(3, 7, n) * 0.3, 0, 1
        )

        # ── Labor & supply ────────────────────────────────────────────────────
        d["labor_availability"]        = np.clip(0.6 + rng.beta(4, 6, n) * 0.4, 0, 1)
        d["storage_access"]            = rng.binomial(
            1, np.clip(0.2 + 0.3 * d["mechanization_level"], 0, 1), n
        ).astype(float)
        d["supply_chain_integration"]  = np.clip(
            0.2 + 0.5 * d["cooperative_membership"] + rng.beta(2, 8, n) * 0.3, 0, 1
        )
        d["disaster_preparedness"]     = np.clip(
            0.2 + 0.3 * d["insurance_coverage"]
            + 0.3 * d["education_level"] / 5 + rng.beta(2, 8, n) * 0.2, 0, 1
        )
        d["alternative_income_sources"]= np.clip(
            0.3 + 0.3 * d["diversification_index"] + rng.beta(3, 7, n) * 0.4, 0, 1
        )
        d["livestock_ownership"]       = rng.binomial(
            1, np.clip(0.5 + 0.2 * (d["land_size"] / 5), 0, 1), n
        ).astype(float)

        # ── Behavioral (derived in _finalize) ─────────────────────────────────
        d["seasonal_payment_consistency"] = np.zeros(n)
        d["repayment_velocity_proxy"]     = np.zeros(n)
        d["climate_debt_compound_stress"] = np.zeros(n)

        # ── Assemble & finalize ────────────────────────────────────────────────
        out = pd.DataFrame({"farmer_id": farmer_ids})
        for feat in FEATURE_NAMES:
            out[feat] = d[feat]

        # Default label (overwritten by HybridDataLoader when used in training)
        default_p     = self._compute_default_probability(d, annual_income, n)
        out["default_flag"] = rng.binomial(1, default_p, n).astype(int)

        out = self._finalize_derived(out)

        # ── Final shape assertion ─────────────────────────────────────────────
        feat_cols = [c for c in out.columns if c in set(FEATURE_NAMES)]
        assert len(feat_cols) == INPUT_DIM, (
            f"DatasetGenerator: generated {len(feat_cols)} feature columns, "
            f"expected INPUT_DIM={INPUT_DIM}"
        )
        return out

    @staticmethod
    def _compute_default_probability(d: dict, annual_income: np.ndarray, n: int) -> np.ndarray:
        """Causal default probability model for realistic label generation."""
        fin_stress    = np.clip(d["debt_to_income_ratio"] + d["loan_to_land_ratio"] / 2, 0, 2)
        weather_shock = d["drought_risk_7days"] * 0.6 + d["frost_risk_7days"] * 0.4
        market_shock  = d["price_volatility"] * 0.4 + np.clip(-d["price_trend"], 0, 0.3)
        ndvi_penalty  = np.clip((0.4 - d["ndvi_current"]), 0, 0.4) * 0.3
        rbi_pressure  = np.clip((d["rbi_repo_rate"] - 6.5) / 10, 0, 0.05)
        prot          = (d["insurance_coverage"] * 0.15
                         + d["irrigation_access"] * 0.10
                         + (d["education_level"] - 1) / 20)
        pay_hist      = (1.0 - d["payment_history_score"]) * 0.30
        raw = (0.05
               + fin_stress    * 0.25
               + weather_shock * 0.20
               + market_shock  * 0.15
               + ndvi_penalty
               + rbi_pressure
               + pay_hist
               - prot)
        return np.clip(raw, 0.01, 0.60)

    @staticmethod
    def _finalize_derived(df: pd.DataFrame) -> pd.DataFrame:
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
#  SECTION 3 — METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    return float(ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic)


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(2 * roc_auc_score(y_true, y_prob) - 1)


def full_scorecard(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> dict:
    return {
        "label":        label,
        "auc":          float(roc_auc_score(y_true, y_prob)),
        "ks":           ks_statistic(y_true, y_prob),
        "gini":         gini_coefficient(y_true, y_prob),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier":        float(brier_score_loss(y_true, y_prob)),
        "default_rate": float(y_true.mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PDO SCORE TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

class PDOScoreTransform:
    """
    Industry-standard Points-to-Double-Odds (PDO) transform.
    Default: score=300 at odds=1:1, doubles every 50 points.
    Maps ML probability → 300–850 credit score.
    """
    BANDS = [
        (750, 850, "A+",  "Approve — Premium",   "Base Rate"),
        (700, 749, "A",   "Approve",              "Base Rate + 0.5%"),
        (650, 699, "B+",  "Approve — Monitored",  "Base Rate + 1.5%"),
        (600, 649, "B",   "Approve — Conditional","Base Rate + 2.5%"),
        (550, 599, "C",   "Refer",                "Base Rate + 4.0%"),
        (500, 549, "D",   "Decline — Review",     "Base Rate + 6.0%"),
        (  0, 499, "E",   "Decline",              "N/A"),
    ]

    def __init__(self, base_score: int = 300, pdo: int = 50, base_odds: float = 1.0):
        self.base_score = base_score
        self.pdo        = pdo
        self.base_odds  = base_odds
        self.factor     = pdo / np.log(2)
        self.offset     = base_score - self.factor * np.log(base_odds)

    def transform(self, pd_estimate: float) -> int:
        pd_clipped = float(np.clip(pd_estimate, 1e-6, 1 - 1e-6))
        log_odds   = np.log((1 - pd_clipped) / pd_clipped)
        raw_score  = self.offset + self.factor * log_odds
        return int(np.clip(round(raw_score), 300, 850))

    def score_band(self, score: int) -> dict:
        for lo, hi, band, decision, rate in self.BANDS:
            if lo <= score <= hi:
                return {"band": band, "decision": decision, "interest_tier": rate}
        return {"band": "E", "decision": "Decline", "interest_tier": "N/A"}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — POLICY ADJUSTMENT LAYER
# ══════════════════════════════════════════════════════════════════════════════

class PolicyAdjustmentLayer:
    """
    RBI / NABARD regulatory rule-based overrides applied on top of the ML score.
    These replicate typical Indian agricultural credit policy hard-stops.
    """

    def apply(self, features: dict, ml_score: int) -> Tuple[int, List[str]]:
        """
        Returns (adjusted_score, list_of_applied_policy_codes).
        Adjustments are capped so the score never moves by more than ±75 points.
        """
        score    = ml_score
        applied  = []

        # P1: High drought risk + no irrigation → –30
        if features.get("drought_risk_7days", 0) > 0.70 \
                and features.get("irrigation_access", 1) < 0.5:
            score -= 30; applied.append("P1_DROUGHT_NO_IRRIGATION")

        # P2: Payment history ≥ 0.90 → +25
        if features.get("payment_history_score", 0) >= 0.90:
            score += 25; applied.append("P2_EXCELLENT_PAYMENT_HISTORY")

        # P3: Debt-to-income > 1.0 → –40
        if features.get("debt_to_income_ratio", 0) > 1.0:
            score -= 40; applied.append("P3_HIGH_DTI")

        # P4: Insurance + cooperative → +15
        if features.get("insurance_coverage", 0) > 0.5 \
                and features.get("cooperative_membership", 0) > 0.5:
            score += 15; applied.append("P4_INSURED_COOP_MEMBER")

        # P5: NDVI anomaly severely negative → –20
        if features.get("ndvi_anomaly", 0) < -1.5:
            score -= 20; applied.append("P5_NDVI_SEVERE_ANOMALY")

        # P6: RBI repo rate elevated (> 7.5%) → –10
        if features.get("rbi_repo_rate", 0) > 7.5:
            score -= 10; applied.append("P6_HIGH_REPO_RATE")

        score = int(np.clip(score, ml_score - 75, ml_score + 75))
        score = int(np.clip(score, 300, 850))
        return score, applied


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — UNIFIED SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedSHAPExplainer:
    """
    Wraps shap.TreeExplainer for the XGBoost base estimator.
    Produces feature importance, top adverse codes, and bootstrap stability.
    """

    DISPLAY_NAMES = {
        "climate_debt_compound_stress": "Climate-Financial Compound Stress",
        "debt_to_income_ratio":         "Debt-to-Income Ratio",
        "payment_history_score":        "Payment History",
        "drought_risk_7days":           "7-Day Drought Risk",
        "ndvi_current":                 "NDVI (Vegetation Health)",
        "ndvi_anomaly":                 "NDVI Anomaly",
        "rbi_repo_rate":                "RBI Repo Rate",
        "rbi_wpi_inflation":            "WPI Inflation",
        "loan_to_land_ratio":           "Loan-to-Land Ratio",
        "soil_moisture_index":          "Soil Moisture Index",
        "price_volatility":             "Crop Price Volatility",
        "irrigation_access":            "Irrigation Access",
    }

    def __init__(self, xgb_model, feature_names: List[str],
                 background_data: Optional[pd.DataFrame] = None):
        self.feature_names = feature_names
        self._explainer    = shap.TreeExplainer(xgb_model)
        self._bg           = background_data

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    def feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv   = self.explain(X)
        mean_abs = np.abs(sv).mean(axis=0)
        df   = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": mean_abs,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["display_name"] = df["feature"].map(self.DISPLAY_NAMES).fillna(df["feature"])
        return df

    def feature_stability(self, X: pd.DataFrame, n_bootstrap: int = 20) -> pd.DataFrame:
        """Bootstrap SHAP importance stability (coefficient of variation per feature)."""
        rng     = np.random.default_rng(42)
        n       = len(X)
        records = []
        for _ in range(n_bootstrap):
            idx  = rng.choice(n, size=n, replace=True)
            sv   = self.explain(X.iloc[idx])
            mean_abs = np.abs(sv).mean(axis=0)
            records.append(mean_abs)
        arr = np.array(records)          # (n_bootstrap, n_features)
        df  = pd.DataFrame({
            "feature":       self.feature_names,
            "mean_imp":      arr.mean(axis=0),
            "std_imp":       arr.std(axis=0),
            "cv":            arr.std(axis=0) / (arr.mean(axis=0) + 1e-9),
        }).sort_values("mean_imp", ascending=False).reset_index(drop=True)
        df["stable"] = df["cv"] < 0.25
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — BENCHMARK SUITE
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Runs five models on the same train/test split for paper Table 2.
    Models:
      (a) Logistic Regression (traditional scorecard baseline)
      (b) WoE scorecard proxy (binned LR)
      (c) XGBoost standalone
      (d) LightGBM standalone
      (e) AgroScore stacking ensemble (proposed method)
    """

    @staticmethod
    def run(X_train: pd.DataFrame, y_train: np.ndarray,
            X_test: pd.DataFrame,  y_test: np.ndarray,
            pos_w: float) -> pd.DataFrame:

        rows = []
        models_to_run = [
            ("Logistic Regression",    "lr",    False),
            ("WoE Scorecard (proxy)",  "woe",   False),
            ("XGBoost Standalone",     "xgb_s", False),
            ("LightGBM Standalone",    "lgb_s", False),
        ]
        for label, key, _ in models_to_run:
            try:
                row = BenchmarkSuite._fit_eval(
                    X_train, y_train, X_test, y_test, pos_w, key, label
                )
                rows.append(row)
            except Exception as e:
                rows.append({"label": label, "auc": 0, "ks": 0, "gini": 0,
                             "avg_precision": 0, "brier": 1, "error": str(e)})
        return pd.DataFrame(rows)

    @staticmethod
    def _fit_eval(Xtr, ytr, Xte, yte, pos_w: float,
                  key: str, label: str) -> dict:
        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(Xtr)
        Xte_s  = scaler.transform(Xte)

        if key == "lr":
            clf = LogisticRegression(
                C=0.1, solver="lbfgs", max_iter=500,
                class_weight="balanced", random_state=42
            )
            clf.fit(Xtr_s, ytr)
            probs = clf.predict_proba(Xte_s)[:, 1]

        elif key == "woe":
            disc  = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
            Xtr_b = disc.fit_transform(Xtr_s)
            Xte_b = disc.transform(Xte_s)
            clf   = LogisticRegression(
                C=0.5, solver="lbfgs", max_iter=500,
                class_weight="balanced", random_state=42
            )
            clf.fit(Xtr_b, ytr)
            probs = clf.predict_proba(Xte_b)[:, 1]

        elif key == "xgb_s":
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.80,
                scale_pos_weight=pos_w, random_state=42,
                eval_metric="logloss", verbosity=0
            )
            clf.fit(Xtr, ytr)
            probs = clf.predict_proba(Xte)[:, 1]

        elif key == "lgb_s":
            clf = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.80,
                scale_pos_weight=pos_w, random_state=42,
                verbosity=-1
            )
            clf.fit(Xtr, ytr)
            probs = clf.predict_proba(Xte)[:, 1]

        else:
            raise ValueError(f"Unknown benchmark key: {key}")

        return {k: round(v, 4) for k, v in full_scorecard(yte, probs, label).items()
                if k != "default_rate"}

    @staticmethod
    def to_latex(df: pd.DataFrame) -> str:
        return df.to_latex(
            index=False, float_format="%.4f",
            caption="Benchmark comparison: AUC, KS, Gini across five models.",
            label="tab:benchmark",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

class AblationStudy:
    """
    Six controlled ablation experiments measuring Δ AUC / KS / Gini vs. full model.
    """
    EXPERIMENTS = {
        "full":              [],
        "no_climate":        ["current_temperature", "current_humidity", "temperature_stress",
                              "humidity_stress", "drought_risk_7days", "frost_risk_7days",
                              "excess_rain_risk", "soil_moisture_index",
                              "ndvi_current", "ndvi_anomaly", "seasonal_rainfall_deviation"],
        "no_financial":      ["debt_to_income_ratio", "loan_to_land_ratio",
                              "payment_history_score", "savings_to_income_ratio",
                              "credit_utilization", "log_debt_to_income", "log_loan_to_land"],
        "no_behavioral":     ["seasonal_payment_consistency", "repayment_velocity_proxy",
                              "climate_debt_compound_stress"],
        "no_rbi_macro":      ["rbi_repo_rate", "rbi_wpi_inflation"],
        "no_ndvi":           ["ndvi_current", "ndvi_anomaly"],
    }

    def run(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_test:  pd.DataFrame, y_test:  np.ndarray,
            pos_w: float) -> pd.DataFrame:
        rows     = []
        base_auc = base_ks = base_gini = None

        for exp_name, drop_cols in self.EXPERIMENTS.items():
            keep = [f for f in FEATURE_NAMES if f not in drop_cols]
            if not keep:
                continue
            try:
                Xtr_e = X_train[keep]
                Xte_e = X_test[keep]
                clf   = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.80, colsample_bytree=0.80,
                    scale_pos_weight=pos_w, random_state=42,
                    eval_metric="logloss", verbosity=0
                )
                clf.fit(Xtr_e, y_train)
                probs = clf.predict_proba(Xte_e)[:, 1]
                m = full_scorecard(y_test, probs, exp_name)

                if exp_name == "full":
                    base_auc  = m["auc"]
                    base_ks   = m["ks"]
                    base_gini = m["gini"]

                rows.append({
                    "experiment":   exp_name,
                    "n_features":   len(keep),
                    "auc":          round(m["auc"],  4),
                    "ks":           round(m["ks"],   4),
                    "gini":         round(m["gini"], 4),
                    "Δauc":         round(m["auc"]  - (base_auc  or m["auc"]),  4),
                    "Δks":          round(m["ks"]   - (base_ks   or m["ks"]),   4),
                    "Δgini":        round(m["gini"] - (base_gini or m["gini"]), 4),
                })
            except Exception as e:
                rows.append({"experiment": exp_name, "error": str(e)})

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FULL FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════════════

class FullFairnessAudit:
    """
    Five fairness metrics with optional instance-reweighting mitigation.
    Groups = geographic regions (proxy for socioeconomic subgroups).
    """

    THRESHOLDS = {
        "equalised_odds_diff":     0.05,
        "demographic_parity_diff": 0.10,
        "predictive_parity_diff":  0.10,
        "max_brier_delta":         0.03,
        "max_wasserstein":         0.10,
    }

    def audit(self, y_true: np.ndarray, y_prob: np.ndarray,
              groups: np.ndarray, threshold: float = 0.50) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        report = {}

        # 1. Equalised Odds Difference
        try:
            from fairlearn.metrics import equalized_odds_difference
            report["equalised_odds_diff"] = float(equalized_odds_difference(
                y_true=y_true, y_pred=y_pred, sensitive_features=groups
            ))
        except Exception:
            report["equalised_odds_diff"] = self._manual_eod(y_true, y_pred, groups)

        # 2. Demographic Parity Difference
        group_approval = {
            g: float(y_pred[groups == g].mean()) for g in np.unique(groups)
        }
        vals = list(group_approval.values())
        report["demographic_parity_diff"] = float(max(vals) - min(vals))
        report["approval_rate_by_group"]  = group_approval

        # 3. Predictive Parity Difference
        group_precision = {}
        for g in np.unique(groups):
            mask = groups == g
            if y_pred[mask].sum() > 0:
                group_precision[g] = float(
                    precision_score(y_true[mask], y_pred[mask], zero_division=0)
                )
        prec_vals = list(group_precision.values())
        report["predictive_parity_diff"] = (
            float(max(prec_vals) - min(prec_vals)) if prec_vals else 0.0
        )
        report["precision_by_group"] = group_precision

        # 4. Calibration within groups (Brier score deviation)
        overall_brier = float(brier_score_loss(y_true, y_prob))
        group_brier   = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 10:
                group_brier[g] = float(brier_score_loss(y_true[mask], y_prob[mask]))
        brier_vals = list(group_brier.values())
        report["max_brier_delta"] = (
            float(max(abs(b - overall_brier) for b in brier_vals)) if brier_vals else 0.0
        )
        report["brier_by_group"] = group_brier

        # 5. Wasserstein distance between group score distributions
        group_scores = {g: y_prob[groups == g] for g in np.unique(groups)}
        glist  = list(group_scores.keys())
        wdists = [
            wasserstein_distance(group_scores[glist[i]], group_scores[glist[j]])
            for i in range(len(glist)) for j in range(i + 1, len(glist))
        ]
        report["max_wasserstein"] = float(max(wdists)) if wdists else 0.0

        report["pass_fail"] = {
            k: ("PASS" if report[k] < v else "FAIL ⚠")
            for k, v in self.THRESHOLDS.items() if k in report
        }
        self._print_report(report)
        return report

    @staticmethod
    def _manual_eod(y_true, y_pred, groups) -> float:
        tpr_by_g, fpr_by_g = {}, {}
        for g in np.unique(groups):
            m  = groups == g
            tp = ((y_pred[m] == 1) & (y_true[m] == 1)).sum()
            fn = ((y_pred[m] == 0) & (y_true[m] == 1)).sum()
            fp = ((y_pred[m] == 1) & (y_true[m] == 0)).sum()
            tn = ((y_pred[m] == 0) & (y_true[m] == 0)).sum()
            tpr_by_g[g] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_by_g[g] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(max(
            max(tpr_by_g.values()) - min(tpr_by_g.values()),
            max(fpr_by_g.values()) - min(fpr_by_g.values())
        ))

    @staticmethod
    def _print_report(r: dict) -> None:
        print("\n" + "─" * 60)
        print("  FULL FAIRNESS AUDIT")
        print("─" * 60)
        for k in ["equalised_odds_diff", "demographic_parity_diff",
                  "predictive_parity_diff", "max_brier_delta", "max_wasserstein"]:
            if k in r:
                flag = r["pass_fail"].get(k, "")
                print(f"  {k:<35}: {r[k]:.4f}  {flag}")
        print("─" * 60)

    def reweigh_by_group(self, y_train: np.ndarray,
                         groups_train: np.ndarray) -> np.ndarray:
        """
        Returns instance weights that upweight under-represented default events
        per group, implementing the reweighing bias-mitigation strategy.
        """
        weights = np.ones(len(y_train))
        for g in np.unique(groups_train):
            m    = groups_train == g
            pos  = y_train[m].sum()
            neg  = (1 - y_train[m]).sum()
            total = pos + neg
            if pos > 0 and neg > 0:
                w_pos = total / (2 * pos)
                w_neg = total / (2 * neg)
                weights[m & (y_train == 1)] = w_pos
                weights[m & (y_train == 0)] = w_neg
        return weights


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — HPO OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

class HPOObjective:
    """Optuna objective for joint XGBoost + LightGBM hyperparameter search."""

    def __init__(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
                 X_val: pd.DataFrame, y_val: np.ndarray,
                 pos_w: float):
        self.X_tr  = X_tr
        self.y_tr  = y_tr
        self.X_val = X_val
        self.y_val = y_val
        self.pos_w = pos_w

    def __call__(self, trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model", ["xgb", "lgbm"])
        if model_type == "xgb":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators",   100, 600),
                "max_depth":        trial.suggest_int("max_depth",       3,   8),
                "learning_rate":    trial.suggest_float("lr",            0.01, 0.20, log=True),
                "subsample":        trial.suggest_float("subsample",     0.60, 1.00),
                "colsample_bytree": trial.suggest_float("colsample",     0.50, 1.00),
                "reg_alpha":        trial.suggest_float("alpha",         1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("lambda",        1e-4, 10.0, log=True),
                "scale_pos_weight": self.pos_w,
                "random_state":     42,
                "eval_metric":      "logloss",
                "verbosity":        0,
            }
            clf = xgb.XGBClassifier(**params)
        else:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators",   100, 600),
                "max_depth":        trial.suggest_int("max_depth",       3,   8),
                "learning_rate":    trial.suggest_float("lr",            0.01, 0.20, log=True),
                "subsample":        trial.suggest_float("subsample",     0.60, 1.00),
                "colsample_bytree": trial.suggest_float("colsample",     0.50, 1.00),
                "reg_alpha":        trial.suggest_float("alpha",         1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("lambda",        1e-4, 10.0, log=True),
                "scale_pos_weight": self.pos_w,
                "random_state":     42,
                "verbosity":        -1,
            }
            clf = lgb.LGBMClassifier(**params)

        clf.fit(self.X_tr, self.y_tr)
        probs = clf.predict_proba(self.X_val)[:, 1]
        return float(roc_auc_score(self.y_val, probs))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — AGROSCORE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class AgroScoreModel:
    """
    End-to-end agricultural credit scoring model.

    Training:
        model = AgroScoreModel(use_hpo=True, hpo_trials=40)
        metrics = model.train(df)   # df from HybridDataLoader or DatasetGenerator

    Inference:
        result = model.predict(df_validated)  # 1-row DataFrame from FeatureValidator
        # or via pipeline:
        result = pipeline.score(farmer_id)

    Persistence:
        model._save_artefacts("./models")
        model2 = AgroScoreModel.load_from_manifest("./models/agroscore_<ts>_manifest.json")
    """

    MODEL_VERSION = "3.0.0"

    def __init__(self, use_hpo: bool = True, hpo_trials: int = 40,
                 temporal_cv: bool = False):
        self.use_hpo       = use_hpo
        self.hpo_trials    = hpo_trials
        self.temporal_cv   = temporal_cv
        self.feature_names = FEATURE_NAMES     # identical to pipeline.FEATURE_NAMES
        self._trained      = False

        self.score_transform  = PDOScoreTransform()
        self.policy_layer     = PolicyAdjustmentLayer()

        self._test_metrics:    dict = {}
        self._benchmark_table: Optional[pd.DataFrame] = None
        self._ablation_table:  Optional[pd.DataFrame] = None
        self._fairness_report: Optional[dict]         = None

    # ──────────────────────────────────────────────────────────────────────────
    #  11.1  TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame,
              run_benchmarks: bool = True,
              run_ablation:   bool = True,
              run_fairness:   bool = True) -> dict:
        """
        Train on a DataFrame produced by HybridDataLoader or DatasetGenerator.
        Expected columns: FEATURE_NAMES + default_flag (+ optional farmer_id, region).

        Returns a metrics dict.
        """
        assert "default_flag" in df.columns, "Training data must have a 'default_flag' column."

        X = FeatureValidator.validate(df)  # shape (N, 77) — guarantees INPUT_DIM columns
        y = df["default_flag"].values.astype(int)
        assert X.shape[1] == INPUT_DIM, (
            f"AgroScoreModel.train: X has {X.shape[1]} columns, expected INPUT_DIM={INPUT_DIM}"
        )

        groups = df.get("region", pd.Series(["Unknown"] * len(df))).values

        # ── Train / test split ─────────────────────────────────────────────────
        if self.temporal_cv and "disbursed_date" in df.columns:
            split = int(len(X) * 0.80)
            idx   = df["disbursed_date"].argsort().values
            tr, te = idx[:split], idx[split:]
        else:
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(
                np.arange(len(X)), test_size=0.20, stratify=y, random_state=42
            )

        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        g_te       = groups[te]

        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        print(f"  Training set: {len(X_tr)}, Test set: {len(X_te)}, "
              f"Default rate: {y_tr.mean():.2%}, pos_weight: {pos_w:.2f}")

        # ── HPO ────────────────────────────────────────────────────────────────
        xgb_p, lgbm_p = self._get_hpo_params(X_tr, y_tr, pos_w)

        # ── Stack ──────────────────────────────────────────────────────────────
        base = [
            ("xgb",  xgb.XGBClassifier(**xgb_p)),
            ("lgbm", lgb.LGBMClassifier(**lgbm_p)),
            ("rf",   RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            )),
        ]
        meta = CalibratedClassifierCV(
            LogisticRegression(C=0.10, solver="lbfgs", max_iter=1000,
                               class_weight="balanced"),
            method="sigmoid", cv=3
        )
        self.stack = StackingClassifier(
            estimators=base, final_estimator=meta,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            passthrough=True, n_jobs=-1
        )
        self.stack.fit(X_tr, y_tr)

        # ── SHAP explainer (uses XGBoost base learner) ─────────────────────────
        xgb_base      = self.stack.estimators_[0]
        bg_idx        = np.random.default_rng(42).choice(len(X_tr),
                                                          size=min(200, len(X_tr)),
                                                          replace=False)
        self._bg_sample   = X_tr.iloc[bg_idx]
        self.shap_explainer = UnifiedSHAPExplainer(xgb_base, self.feature_names,
                                                   self._bg_sample)
        self.feature_importance = self.shap_explainer.feature_importance(self._bg_sample)

        # ── Test metrics ───────────────────────────────────────────────────────
        probs_te        = self.stack.predict_proba(X_te)[:, 1]
        self._test_metrics = full_scorecard(y_te, probs_te, "AgroScore_Stack")
        print(f"\n  AgroScore Stacking Ensemble:")
        print(f"    AUC   = {self._test_metrics['auc']:.4f}")
        print(f"    KS    = {self._test_metrics['ks']:.4f}")
        print(f"    Gini  = {self._test_metrics['gini']:.4f}")
        print(f"    Brier = {self._test_metrics['brier']:.4f}")

        # ── Research extensions ────────────────────────────────────────────────
        if run_benchmarks:
            print("\n  [Benchmark Suite]")
            bm = BenchmarkSuite.run(X_tr, y_tr, X_te, y_te, pos_w)
            # Append proposed model row
            proposed_row = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in self._test_metrics.items()
                            if k != "default_rate"}
            proposed_row["label"] = "AgroScore Ensemble (proposed)"
            self._benchmark_table = pd.concat(
                [bm, pd.DataFrame([proposed_row])], ignore_index=True
            )
            print(self._benchmark_table[["label", "auc", "ks", "gini"]].to_string(index=False))

        if run_ablation:
            print("\n  [Ablation Study]")
            self._ablation_table = AblationStudy().run(X_tr, y_tr, X_te, y_te, pos_w)
            print(self._ablation_table[["experiment", "auc", "ks", "Δauc"]].to_string(index=False))

        if run_fairness:
            print("\n  [Full Fairness Audit]")
            self._fairness_report = FullFairnessAudit().audit(y_te, probs_te, g_te)

        self._trained = True
        self._save_artefacts()
        return self._test_metrics

    def _get_hpo_params(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
                        pos_w: float) -> Tuple[dict, dict]:
        """Run Optuna HPO or return sensible defaults."""
        xgb_defaults = dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, scale_pos_weight=pos_w,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        lgbm_defaults = dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80, scale_pos_weight=pos_w,
            random_state=42, verbosity=-1
        )
        if not self.use_hpo or self.hpo_trials < 1:
            return xgb_defaults, lgbm_defaults

        from sklearn.model_selection import train_test_split
        X_hpo_tr, X_hpo_val, y_hpo_tr, y_hpo_val = train_test_split(
            X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=99
        )
        objective = HPOObjective(X_hpo_tr, y_hpo_tr, X_hpo_val, y_hpo_val, pos_w)
        study     = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.hpo_trials, show_progress_bar=False)

        best = study.best_params
        if best.get("model") == "xgb":
            xgb_p = dict(
                n_estimators=best["n_estimators"], max_depth=best["max_depth"],
                learning_rate=best["lr"], subsample=best["subsample"],
                colsample_bytree=best["colsample"],
                reg_alpha=best["alpha"], reg_lambda=best["lambda"],
                scale_pos_weight=pos_w, random_state=42,
                eval_metric="logloss", verbosity=0
            )
            lgbm_p = lgbm_defaults
        else:
            lgbm_p = dict(
                n_estimators=best["n_estimators"], max_depth=best["max_depth"],
                learning_rate=best["lr"], subsample=best["subsample"],
                colsample_bytree=best["colsample"],
                reg_alpha=best["alpha"], reg_lambda=best["lambda"],
                scale_pos_weight=pos_w, random_state=42, verbosity=-1
            )
            xgb_p  = xgb_defaults

        print(f"  HPO best AUC: {study.best_value:.4f}  model={best.get('model')}")
        return xgb_p, lgbm_p

    # ──────────────────────────────────────────────────────────────────────────
    #  11.2  INFERENCE
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Single-row inference.  df must be validated (shape 1×77, dtype float64).
        Returns a dict with pd_estimate, ml_score, agro_score, risk_band,
                          decision, interest_tier, top_adverse_code, policy_applied.
        """
        assert self._trained, "Call train() or load_from_manifest() first."
        X     = FeatureValidator.validate(df)
        assert X.shape[1] == INPUT_DIM, (
            f"predict(): X has {X.shape[1]} cols, expected {INPUT_DIM}"
        )
        prob   = float(self.stack.predict_proba(X)[0, 1])
        ml_sc  = self.score_transform.transform(prob)
        feats  = {f: float(X.iloc[0][f]) for f in self.feature_names}
        fin_sc, applied = self.policy_layer.apply(feats, ml_sc)
        band   = self.score_transform.score_band(fin_sc)

        # Top adverse SHAP code
        sv     = self.shap_explainer.explain(X)[0]
        fa     = np.array(self.feature_names)
        dm     = self.shap_explainer.DISPLAY_NAMES
        top_adv = ""
        if (sv > 0).any():
            top_idx = int(np.argmax(np.where(sv > 0, sv, -np.inf)))
            top_adv = dm.get(fa[top_idx], fa[top_idx])

        return {
            "pd_estimate":   round(prob, 6),
            "ml_score":      ml_sc,
            "agro_score":    fin_sc,
            "risk_band":     band["band"],
            "decision":      band["decision"],
            "interest_tier": band["interest_tier"],
            "top_adverse_code": top_adv,
            "policy_applied":  "; ".join(applied) if applied else "",
        }

    def batch_predict(self, df: pd.DataFrame,
                      id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Batch inference for a DataFrame with FEATURE_NAMES columns.
        Returns DataFrame with one result row per input row.
        """
        assert self._trained
        id_s  = (df[id_col].reset_index(drop=True) if id_col and id_col in df.columns
                 else pd.Series(range(len(df)), name="farmer_id"))
        X     = FeatureValidator.validate(df)
        probs = self.stack.predict_proba(X)[:, 1]
        sv    = self.shap_explainer.explain(X)
        fa    = np.array(self.feature_names)
        dm    = self.shap_explainer.DISPLAY_NAMES
        rows  = []
        for i, p in enumerate(probs):
            ml_sc  = self.score_transform.transform(float(p))
            feats  = {f: float(X.iloc[i][f]) for f in self.feature_names}
            fin_sc, applied = self.policy_layer.apply(feats, ml_sc)
            band   = self.score_transform.score_band(fin_sc)
            sv_i   = sv[i]
            top_adv = ""
            if (sv_i > 0).any():
                top_idx = int(np.argmax(np.where(sv_i > 0, sv_i, -np.inf)))
                top_adv = dm.get(fa[top_idx], fa[top_idx])
            rows.append({
                "pd_estimate":    round(float(p), 6),
                "ml_score":       ml_sc,
                "agro_score":     fin_sc,
                "risk_band":      band["band"],
                "decision":       band["decision"],
                "interest_tier":  band["interest_tier"],
                "top_adverse_code": top_adv,
                "policy_applied": "; ".join(applied) if applied else "",
            })
        result = pd.DataFrame(rows)
        result.insert(0, id_s.name or "farmer_id", id_s.values)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    #  11.3  POST-TRAIN RESEARCH METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def feature_stability_report(self, X_test: pd.DataFrame,
                                  n_bootstrap: int = 20) -> pd.DataFrame:
        assert self._trained
        return self.shap_explainer.feature_stability(
            FeatureValidator.validate(X_test), n_bootstrap
        )

    def decision_consistency_check(self, X_test: pd.DataFrame,
                                    n_samples: int = 50) -> dict:
        """
        Score each farmer twice with tiny Gaussian noise (ε=0.001).
        Consistent if both scorings land in the same risk band.
        """
        assert self._trained
        X    = FeatureValidator.validate(X_test).head(n_samples)
        Xn   = (X + np.random.default_rng(42).normal(0, 0.001, X.shape)).clip(lower=0)
        b1   = self.batch_predict(X)["risk_band"].values
        b2   = self.batch_predict(Xn)["risk_band"].values
        rate = float((b1 == b2).mean())
        print(f"\n  Decision consistency: {rate:.2%}  ({int(rate * n_samples)}/{n_samples})")
        return {"consistency_rate": rate, "n_tested": n_samples,
                "n_consistent": int(rate * n_samples)}

    def benchmark_table(self) -> Optional[pd.DataFrame]:
        return self._benchmark_table

    def ablation_table(self) -> Optional[pd.DataFrame]:
        return self._ablation_table

    def fairness_report(self) -> Optional[dict]:
        return self._fairness_report

    def latex_tables(self) -> str:
        out = []
        if self._benchmark_table is not None:
            out.append("% Table 2: Benchmark Comparison")
            out.append(BenchmarkSuite.to_latex(self._benchmark_table))
        if self._ablation_table is not None:
            out.append("% Table 3: Ablation Study")
            out.append(
                self._ablation_table[["experiment", "auc", "ks", "gini",
                                       "Δauc", "Δks", "Δgini"]].to_latex(
                    float_format="%.4f",
                    caption="Ablation study results. Δ columns show degradation vs. full model.",
                    label="tab:ablation",
                )
            )
        return "\n\n".join(out)

    # ──────────────────────────────────────────────────────────────────────────
    #  11.4  PERSISTENCE
    # ──────────────────────────────────────────────────────────────────────────

    def _save_artefacts(self, output_dir: str = ".") -> None:
        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{output_dir}/agroscore_{ts}"
        paths = {
            "stack":           f"{base}_stack.pkl",
            "feature_names":   f"{base}_features.pkl",
            "score_transform": f"{base}_pdo.pkl",
            "bg_sample":       f"{base}_bg.pkl",
            "feature_imp":     f"{base}_importance.csv",
        }
        joblib.dump(self.stack,           paths["stack"])
        joblib.dump(self.feature_names,   paths["feature_names"])
        joblib.dump(self.score_transform, paths["score_transform"])
        joblib.dump(self._bg_sample,      paths["bg_sample"])
        self.feature_importance.to_csv(paths["feature_imp"], index=False)
        manifest = {
            "version":       self.MODEL_VERSION,
            "timestamp":     ts,
            "input_dim":     INPUT_DIM,
            "feature_count": len(self.feature_names),
            "test_metrics":  self._test_metrics,
            "files":         paths,
        }
        manifest_path = f"{base}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  Artefacts → {base}_*.pkl/.csv")
        print(f"  Manifest  → {manifest_path}")
        self._manifest_path = manifest_path

    @classmethod
    def load_from_manifest(cls, manifest_path: str) -> "AgroScoreModel":
        with open(manifest_path) as f:
            manifest = json.load(f)
        files = manifest["files"]

        # Schema-drift guard
        saved_dim = manifest.get("input_dim", manifest.get("feature_count"))
        if saved_dim and int(saved_dim) != INPUT_DIM:
            raise RuntimeError(
                f"Schema drift: saved INPUT_DIM={saved_dim}, current={INPUT_DIM}. Retrain."
            )

        obj                 = cls.__new__(cls)
        obj.stack           = joblib.load(files["stack"])
        obj.feature_names   = joblib.load(files["feature_names"])
        obj.score_transform = joblib.load(files["score_transform"])
        obj.policy_layer    = PolicyAdjustmentLayer()
        obj._test_metrics   = manifest.get("test_metrics", {})
        obj._trained        = True
        obj.use_hpo         = False
        obj.hpo_trials      = 0
        obj.temporal_cv     = False
        obj.feature_importance = pd.read_csv(files["feature_imp"])
        obj._bg_sample      = joblib.load(files["bg_sample"])
        obj.shap_explainer  = UnifiedSHAPExplainer(
            obj.stack.estimators_[0], obj.feature_names, obj._bg_sample
        )
        obj._benchmark_table = None
        obj._ablation_table  = None
        obj._fairness_report = None

        if list(obj.feature_names) != FEATURE_NAMES:
            diff = set(obj.feature_names).symmetric_difference(set(FEATURE_NAMES))
            raise RuntimeError(f"Feature name drift — retrain. Diff: {diff}")

        print(f"  Model loaded (v{manifest['version']}, INPUT_DIM={INPUT_DIM})")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — full integration test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  AgroScore v3.0 — Refactored Integration Test")
    print(f"  INPUT_DIM (from pipeline): {INPUT_DIM}")
    print("=" * 70)

    print("\n[1/6] Schema alignment check...")
    assert check_schema_alignment(verbose=True), "Schema alignment FAILED."

    print("\n[2/6] Building hybrid dataset...")
    loader = HybridDataLoader()
    gen    = DatasetGenerator()
    df     = loader.load(
        real_path      = os.getenv("REAL_LOAN_DATA_PATH", ""),
        synthetic_gen  = gen,
        n_synthetic    = 6000,
    )
    print(f"      Shape        : {df.shape}")
    print(f"      Default rate : {df['default_flag'].mean():.2%}")
    # Confirm feature count matches INPUT_DIM
    feat_cols = [c for c in df.columns if c in set(FEATURE_NAMES)]
    assert len(feat_cols) == INPUT_DIM, (
        f"Dataset feature column count {len(feat_cols)} != INPUT_DIM {INPUT_DIM}"
    )
    print(f"      Feature cols : {len(feat_cols)} ✓  (== INPUT_DIM={INPUT_DIM})")

    print("\n[3/6] Training (benchmarks + ablation + fairness)...")
    model   = AgroScoreModel(use_hpo=True, hpo_trials=30, temporal_cv=False)
    metrics = model.train(df, run_benchmarks=True, run_ablation=True, run_fairness=True)

    print("\n[4/6] Feature stability report (bootstrap SHAP):")
    X_sample = df[FEATURE_NAMES].sample(300, random_state=0)
    stability = model.feature_stability_report(X_sample, n_bootstrap=15)
    print(stability.head(10).to_string(index=False))

    print("\n[5/6] Decision consistency check:")
    model.decision_consistency_check(X_sample, n_samples=50)

    print("\n[6/6] LaTeX tables for paper:")
    print(model.latex_tables()[:600], "...\n[truncated]")

    print("\n" + "=" * 70)
    print("  v3.0 complete. INPUT_DIM contract verified end-to-end.")
    print("=" * 70)
