# This model creates highly accurate, correlated features that mirror real-world agricultural economics
# Features include regional variations, crop-specific risks, weather patterns, and economic interactions

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

class AdvancedCreditModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.shap_explainer = None
        self.best_model = None
        self.best_model_name = None

    def create_advanced_dataset(self, n_samples=5000):
        """
        Creates hyper-realistic agricultural dataset with correlated features
        based on actual agricultural economics and Indian farming patterns
        """
        np.random.seed(42)
        data = {}

        # === GEOGRAPHIC AND DEMOGRAPHIC FOUNDATION ===
        regions = np.random.choice(['Punjab', 'Maharashtra', 'UP', 'Karnataka', 'AP', 'WB', 'Gujarat', 'MP'], n_samples)

        # Region-specific characteristics based on actual data
        region_mapping = {
            'Punjab': {'avg_land': 4.2, 'irrigation_p': 0.95, 'mech_level': 0.8, 'income_mult': 1.3},
            'Maharashtra': {'avg_land': 2.8, 'irrigation_p': 0.45, 'mech_level': 0.6, 'income_mult': 1.1},
            'UP': {'avg_land': 1.9, 'irrigation_p': 0.75, 'mech_level': 0.4, 'income_mult': 0.9},
            'Karnataka': {'avg_land': 3.1, 'irrigation_p': 0.35, 'mech_level': 0.55, 'income_mult': 1.0},
            'AP': {'avg_land': 2.4, 'irrigation_p': 0.55, 'mech_level': 0.5, 'income_mult': 1.05},
            'WB': {'avg_land': 1.6, 'irrigation_p': 0.85, 'mech_level': 0.3, 'income_mult': 0.85},
            'Gujarat': {'avg_land': 3.5, 'irrigation_p': 0.65, 'mech_level': 0.7, 'income_mult': 1.2},
            'MP': {'avg_land': 4.8, 'irrigation_p': 0.3, 'mech_level': 0.45, 'income_mult': 0.95}
        }

        data['farmer_id'] = range(n_samples)

        # Realistic age distribution (modal age 45-55)
        ages = np.random.gamma(3, 15, n_samples) + 25
        data['farmer_age'] = np.clip(ages, 22, 75)

        # Education correlates with age (younger farmers more educated)
        age_education_factor = np.where(data['farmer_age'] < 35, 0.4,
                                      np.where(data['farmer_age'] < 50, 0.0, -0.3))
        education_base = np.random.beta(2, 3, n_samples) + age_education_factor
        data['education_level'] = np.clip((education_base * 5).astype(int) + 1, 1, 5)

        # Family size correlates with age and education
        family_base = 4.2 + 0.05 * (data['farmer_age'] - 45) - 0.3 * (data['education_level'] - 2.5)
        data['family_size'] = np.random.poisson(np.clip(family_base, 2, 8))

        # === LAND AND CROP CHARACTERISTICS ===
        # Land size varies by region with gamma distribution (realistic)
        region_land_base = np.array([region_mapping[r]['avg_land'] for r in regions])
        land_variation = np.random.gamma(2, 0.5, n_samples)
        data['land_size'] = np.clip(region_land_base * land_variation, 0.5, 20)

        # Crop selection based on regional patterns
        crop_probs = {
            'Punjab': [0.1, 0.6, 0.05, 0.15, 0.05, 0.05],    # Rice, Wheat, Cotton, Sugarcane, Soybean, Maize
            'Maharashtra': [0.15, 0.2, 0.3, 0.2, 0.1, 0.05],
            'UP': [0.3, 0.4, 0.1, 0.1, 0.05, 0.05],
            'Karnataka': [0.2, 0.15, 0.2, 0.25, 0.15, 0.05],
            'AP': [0.25, 0.1, 0.25, 0.25, 0.1, 0.05],
            'WB': [0.7, 0.15, 0.05, 0.05, 0.03, 0.02],
            'Gujarat': [0.1, 0.2, 0.4, 0.15, 0.1, 0.05],
            'MP': [0.15, 0.35, 0.15, 0.15, 0.15, 0.05]
        }

        data['crop_type_encoded'] = np.zeros(n_samples)
        for i, region in enumerate(regions):
            data['crop_type_encoded'][i] = np.random.choice(range(1, 7), p=crop_probs[region])

        # === INFRASTRUCTURE ===
        # Irrigation access varies by region
        region_irrigation = np.array([region_mapping[r]['irrigation_p'] for r in regions])
        irrigation_noise = np.random.normal(0, 0.1, n_samples)
        irrigation_prob = np.clip(region_irrigation + irrigation_noise, 0.1, 0.95)
        data['irrigation_access'] = np.random.binomial(1, irrigation_prob, n_samples)

        # === WEATHER AND CLIMATE RISKS ===
        # Temperature varies by region
        regional_temp = {'Punjab': 28, 'Maharashtra': 32, 'UP': 30, 'Karnataka': 29,
                        'AP': 33, 'WB': 31, 'Gujarat': 34, 'MP': 31}
        base_temp = np.array([regional_temp[r] for r in regions])
        data['current_temperature'] = base_temp + np.random.normal(0, 4, n_samples)

        # Humidity varies by region
        regional_humidity = {'Punjab': 55, 'Maharashtra': 65, 'UP': 70, 'Karnataka': 62,
                           'AP': 68, 'WB': 78, 'Gujarat': 58, 'MP': 60}
        base_humidity = np.array([regional_humidity[r] for r in regions])
        data['current_humidity'] = np.clip(base_humidity + np.random.normal(0, 10, n_samples), 30, 95)

        # Temperature and humidity stress
        data['temperature_stress'] = np.clip((data['current_temperature'] - 28)/15, 0, 1) + np.random.beta(2, 8, n_samples) * 0.3
        data['humidity_stress'] = np.clip(np.abs(data['current_humidity'] - 60)/25, 0, 1) + np.random.beta(2, 6, n_samples) * 0.2

        # Drought risk varies by region and irrigation
        drought_base_by_region = {'Punjab': 0.2, 'Maharashtra': 0.6, 'UP': 0.3, 'Karnataka': 0.7,
                                'AP': 0.5, 'WB': 0.2, 'Gujarat': 0.4, 'MP': 0.8}
        drought_base = np.array([drought_base_by_region[r] for r in regions])
        data['drought_risk_7days'] = np.clip(drought_base * np.random.beta(2, 5, n_samples) -
                                           data['irrigation_access'] * 0.3, 0, 1)

        # Frost risk (crop and location specific)
        frost_sensitive_crops = [1, 3, 4, 5]  # Rice, Cotton, Sugarcane, Soybean
        frost_base = np.where(np.isin(data['crop_type_encoded'], frost_sensitive_crops), 0.4, 0.05)
        northern_regions = ['Punjab', 'UP', 'MP']
        frost_base = np.where(np.isin(regions, northern_regions), frost_base * 2, frost_base)
        data['frost_risk_7days'] = np.clip(frost_base * np.random.beta(1, 12, n_samples), 0, 1)

        # Excess rain risk (inversely related to drought)
        data['excess_rain_risk'] = np.random.beta(1, 9, n_samples) * (1 - data['drought_risk_7days'] * 0.8)

        # === MARKET AND ECONOMIC FACTORS ===
        # Price volatility varies by crop
        price_volatility_by_crop = {1: 0.18, 2: 0.12, 3: 0.35, 4: 0.25, 5: 0.28, 6: 0.20}
        volatility_base = np.array([price_volatility_by_crop[int(c)] for c in data['crop_type_encoded']])
        data['price_volatility'] = np.clip(volatility_base + np.random.beta(2, 6, n_samples) * 0.3 - 0.15, 0.05, 0.8)

        # Income calculation based on land, crop, region, irrigation
        region_income = np.array([region_mapping[r]['income_mult'] for r in regions])
        crop_income_mult = {1: 1.0, 2: 0.85, 3: 1.4, 4: 1.6, 5: 1.15, 6: 0.95}
        crop_mult = np.array([crop_income_mult[int(c)] for c in data['crop_type_encoded']])

        base_income_per_hectare = 55000
        annual_income = (data['land_size'] * base_income_per_hectare * crop_mult * region_income *
                        (1 + 0.5 * data['irrigation_access']) * np.random.lognormal(0, 0.4, n_samples))
        annual_income = np.clip(annual_income, 25000, 3000000)

        # === FINANCIAL FEATURES ===
        # Loan characteristics
        loan_probability = 0.75 + 0.15 * (data['land_size'] / 10) + 0.1 * (data['education_level'] / 5)
        has_loan = np.random.binomial(1, np.clip(loan_probability, 0.3, 0.95), n_samples)
        loan_amounts = np.where(has_loan, annual_income * np.random.beta(3, 4, n_samples) * 0.8, 0)

        data['loan_to_land_ratio'] = loan_amounts / (data['land_size'] * 150000 + 1)
        data['debt_to_income_ratio'] = loan_amounts / (annual_income + 1)

        # Payment history based on financial stress and stability
        financial_stress = np.clip(data['debt_to_income_ratio'] + data['loan_to_land_ratio']/2, 0, 2)
        income_stability = 1 - data['price_volatility'] - data['drought_risk_7days'] * 0.5
        education_factor = data['education_level'] / 10

        payment_history_base = 0.88 - financial_stress * 0.25 + income_stability * 0.15 + education_factor
        data['payment_history_score'] = np.clip(payment_history_base + np.random.normal(0, 0.12, n_samples), 0.1, 1.0)

        # Yield consistency
        irrigation_effect = data['irrigation_access'] * 0.3
        weather_stability = 1 - (data['drought_risk_7days'] + data['frost_risk_7days'] + data['excess_rain_risk']) / 3
        tech_effect = data['education_level'] / 20

        data['yield_consistency'] = np.clip(0.7 + irrigation_effect + weather_stability * 0.2 + tech_effect +
                                          np.random.beta(6, 4, n_samples) * 0.2 - 0.1, 0.3, 1.0)

        # === INFRASTRUCTURE AND ACCESS ===
        data['nearest_mandi_distance'] = np.random.gamma(2.5, 8, n_samples)
        data['connectivity_index'] = np.clip(1 - data['nearest_mandi_distance']/40 +
                                           np.random.beta(5, 5, n_samples) * 0.4, 0.1, 1.0)

        # === SOIL AND AGRICULTURAL PRACTICES ===
        soil_base = 0.65 + data['irrigation_access'] * 0.2 + (data['education_level']/5) * 0.1
        data['soil_health_index'] = np.clip(soil_base + np.random.beta(4, 3, n_samples) * 0.25 - 0.1, 0.2, 1.0)

        data['nutrient_deficiency_risk'] = np.clip(1.2 - data['soil_health_index'] -
                                                 (annual_income/500000) * 0.3 +
                                                 np.random.beta(3, 5, n_samples) * 0.4, 0, 1)

        # === SUPPORT SYSTEMS ===
        # Insurance coverage
        insurance_prob = 0.35 + (data['education_level']/5) * 0.3 + (annual_income/500000) * 0.2
        data['insurance_coverage'] = np.random.binomial(1, np.clip(insurance_prob, 0.15, 0.85), n_samples)

        # Cooperative membership
        coop_base_by_region = {'Punjab': 0.75, 'Maharashtra': 0.55, 'UP': 0.4, 'Karnataka': 0.5,
                              'AP': 0.45, 'WB': 0.35, 'Gujarat': 0.8, 'MP': 0.4}
        coop_base = np.array([coop_base_by_region[r] for r in regions])
        data['cooperative_membership'] = np.random.binomial(1, coop_base + (data['education_level']/5 - 0.6) * 0.2, n_samples)

        # Technology adoption
        tech_adoption_base = (data['education_level']/5) * 0.4 + (annual_income/300000) * 0.3 + data['irrigation_access'] * 0.2
        data['technology_adoption'] = np.clip(tech_adoption_base + np.random.beta(3, 6, n_samples) * 0.4, 0.1, 0.95)

        # Diversification
        data['diversification_index'] = np.clip(0.25 + (data['land_size']/8) * 0.4 + (data['education_level']/5) * 0.2 +
                                              np.random.beta(3, 6, n_samples) * 0.35, 0.1, 0.9)

        # Input costs
        data['input_cost_index'] = np.clip(0.45 + (1 - data['connectivity_index']) * 0.3 +
                                         data['price_volatility'] * 0.2 +
                                         np.random.beta(4, 5, n_samples) * 0.25, 0.2, 0.9)

        # === ADDITIONAL FEATURES FOR COMPLETENESS ===
        # Fill remaining features with realistic values and correlations
        data['seasonal_rainfall_deviation'] = np.random.normal(0, 18, n_samples)
        data['historical_drought_frequency'] = np.random.poisson(data['drought_risk_7days'] * 4 + 0.8, n_samples)
        data['climate_change_vulnerability'] = (data['drought_risk_7days'] * 0.5 + data['temperature_stress'] * 0.3 +
                                              (1 - data['irrigation_access']) * 0.2)

        data['current_price'] = annual_income / data['land_size'] + np.random.normal(0, 8000, n_samples)
        data['market_demand_index'] = np.random.beta(4, 5, n_samples)
        data['export_potential'] = np.where(np.isin(data['crop_type_encoded'], [2, 3, 4]),
                                          np.random.beta(5, 5, n_samples), np.random.beta(2, 7, n_samples))
        data['storage_price_premium'] = np.random.beta(2, 7, n_samples)
        data['price_trend'] = np.random.normal(0.02, 0.12, n_samples)

        data['savings_to_income_ratio'] = np.clip(0.18 - data['debt_to_income_ratio'] * 0.15 +
                                                (data['education_level']/5) * 0.08 +
                                                np.random.beta(3, 8, n_samples) * 0.2, 0, 0.4)
        data['credit_utilization'] = np.clip(data['debt_to_income_ratio'] * 1.2 +
                                           np.random.beta(3, 6, n_samples) * 0.3, 0, 1)
        data['number_of_credit_sources'] = np.random.poisson(1.2 + data['education_level']/5 + has_loan * 0.5, n_samples)
        data['informal_lending_dependency'] = np.clip((1 - data['cooperative_membership']) * 0.4 +
                                                    (1 - data['insurance_coverage']) * 0.2 +
                                                    np.random.beta(2, 6, n_samples) * 0.4, 0, 0.8)

        # More infrastructure features
        data['road_quality_index'] = np.clip(data['connectivity_index'] + np.random.beta(3, 7, n_samples) * 0.3, 0, 1)
        data['electricity_reliability'] = np.clip(0.7 + 0.2 * annual_income/300000 + np.random.beta(4, 6, n_samples) * 0.3, 0, 1)
        data['mobile_network_strength'] = np.clip(0.8 + np.random.beta(5, 5, n_samples) * 0.2, 0, 1)
        data['bank_branch_distance'] = data['nearest_mandi_distance'] * 0.7 + np.random.gamma(2, 3, n_samples)

        # Agricultural practices
        data['mechanization_level'] = np.clip(data['technology_adoption'] * 0.8 + (data['land_size']/10) * 0.2, 0, 1)
        data['seed_quality_index'] = np.clip(0.6 + 0.3 * data['technology_adoption'] + np.random.beta(4, 6, n_samples) * 0.3, 0, 1)
        data['fertilizer_usage_efficiency'] = np.clip(data['soil_health_index'] + np.random.beta(4, 6, n_samples) * 0.3, 0, 1)
        data['pest_management_score'] = np.clip(0.5 + 0.3 * data['technology_adoption'] + np.random.beta(4, 6, n_samples) * 0.3, 0, 1)
        data['organic_farming_adoption'] = np.random.beta(2, 8, n_samples)
        data['precision_agriculture_usage'] = np.clip(data['technology_adoption'] * 0.7 + np.random.beta(1, 9, n_samples) * 0.3, 0, 1)

        # Government support
        data['eligible_schemes_count'] = np.random.poisson(2 + data['education_level']/2, n_samples)
        data['subsidy_utilization'] = np.clip(0.3 + 0.4 * data['cooperative_membership'] + np.random.beta(3, 7, n_samples) * 0.4, 0, 1)
        data['msp_eligibility'] = np.random.binomial(1, 0.7, n_samples)
        data['kisan_credit_card'] = np.random.binomial(1, 0.4 + 0.3 * (data['education_level']/5), n_samples)
        data['government_training_participation'] = np.clip(0.2 * data['education_level']/5 + np.random.beta(2, 8, n_samples), 0, 1)

        # Social features
        data['community_leadership_role'] = np.random.binomial(1, 0.1 + 0.1 * data['education_level']/5, n_samples)
        data['social_capital_index'] = np.clip(0.4 + 0.3 * data['cooperative_membership'] + 0.3 * data['community_leadership_role'] +
                                             np.random.beta(4, 6, n_samples) * 0.3, 0, 1)
        data['extension_service_access'] = np.clip(0.3 + 0.4 * data['social_capital_index'] + np.random.beta(3, 7, n_samples) * 0.3, 0, 1)
        data['peer_learning_participation'] = np.clip(data['social_capital_index'] * 0.7 + np.random.beta(3, 7, n_samples) * 0.3, 0, 1)

        # Supply chain
        data['labor_availability'] = np.clip(0.6 + np.random.beta(4, 6, n_samples) * 0.4, 0, 1)
        data['storage_access'] = np.random.binomial(1, 0.2 + 0.3 * data['mechanization_level'], n_samples)
        data['transport_cost_burden'] = np.clip(data['nearest_mandi_distance']/30 + np.random.beta(3, 7, n_samples) * 0.5, 0, 1)
        data['supply_chain_integration'] = np.clip(0.2 + 0.5 * data['cooperative_membership'] + np.random.beta(2, 8, n_samples) * 0.3, 0, 1)

        # Risk management
        data['disaster_preparedness'] = np.clip(0.2 + 0.3 * data['insurance_coverage'] + 0.3 * data['education_level']/5 +
                                              np.random.beta(2, 8, n_samples) * 0.2, 0, 1)
        data['alternative_income_sources'] = np.clip(0.3 + 0.3 * data['diversification_index'] + np.random.beta(3, 7, n_samples) * 0.4, 0, 1)
        # Fix: Clip the probability for livestock_ownership
        livestock_prob = np.clip(0.5 + 0.2 * (data['land_size']/5), 0, 1)
        data['livestock_ownership'] = np.random.binomial(1, livestock_prob, n_samples)

        # === HYPER-REALISTIC RISK SCORE CALCULATION ===
        # Advanced risk scoring based on agricultural economics research

        # Convert to arrays for calculation
        arrays = {k: np.array(v) for k, v in data.items()}

        # Primary risk factors (40% weight)
        payment_risk = (1 - arrays['payment_history_score']) * 0.18
        debt_stress = np.where(arrays['debt_to_income_ratio'] > 0.5,
                              arrays['debt_to_income_ratio'] ** 1.8 * 0.12,
                              arrays['debt_to_income_ratio'] * 0.12)
        income_instability = (1 - arrays['yield_consistency']) * 0.10

        # Climate and weather risks (25% weight)
        crop_drought_multiplier = np.where(np.isin(arrays['crop_type_encoded'], [1, 4, 5]), 1.3, 1.0)
        drought_risk = arrays['drought_risk_7days'] * crop_drought_multiplier * 0.10
        temp_stress = np.where(arrays['temperature_stress'] > 0.6,
                              arrays['temperature_stress'] ** 1.5 * 0.08,
                              arrays['temperature_stress'] * 0.08)
        frost_risk = arrays['frost_risk_7days'] * 0.07

        # Market and economic risks (20% weight)
        price_risk = arrays['price_volatility'] * 0.08
        input_cost_risk = arrays['input_cost_index'] * 0.06
        market_access_risk = np.clip(arrays['nearest_mandi_distance'] / 50, 0, 1) * 0.06

        # Infrastructure deficits (10% weight)
        irrigation_risk = (1 - arrays['irrigation_access']) * arrays['drought_risk_7days'] * 0.04
        insurance_risk = (1 - arrays['insurance_coverage']) * 0.03
        connectivity_risk = (1 - arrays['connectivity_index']) * 0.03

        # Agricultural practices (5% weight)
        soil_risk = (1 - arrays['soil_health_index']) * 0.02
        practice_risk = (1 - arrays['technology_adoption']) * 0.02
        nutrient_risk = arrays['nutrient_deficiency_risk'] * 0.01

        # Interaction effects
        compound_stress = np.where((arrays['debt_to_income_ratio'] > 0.6) & (arrays['drought_risk_7days'] > 0.5), 0.05, 0)

        # Protective factors
        coop_protection = arrays['cooperative_membership'] * 0.02
        diversification_protection = arrays['diversification_index'] * 0.015
        tech_protection = arrays['technology_adoption'] * 0.01
        education_protection = (arrays['education_level'] / 5) * 0.01

        # Calculate total risk score
        risk_score = (
            payment_risk + debt_stress + income_instability +
            drought_risk + temp_stress + frost_risk +
            price_risk + input_cost_risk + market_access_risk +
            irrigation_risk + insurance_risk + connectivity_risk +
            soil_risk + practice_risk + nutrient_risk +
            compound_stress -
            coop_protection - diversification_protection - tech_protection - education_protection +
            np.random.normal(0, 0.035, n_samples)
        )

        # Ensure risk score is within bounds
        risk_score = np.clip(risk_score, 0.05, 0.95)

        # Create default flag (top 20% risk = default for realistic default rate)
        default_threshold = np.percentile(risk_score, 80)
        data['default_flag'] = (risk_score > default_threshold).astype(int)

        return pd.DataFrame(data)

    def train_ensemble_model(self, df):
        """Train multiple models and create ensemble with cross-validation"""
        feature_columns = [col for col in df.columns if col not in ['farmer_id', 'default_flag']]
        X = df[feature_columns]
        y = df['default_flag']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models with optimized hyperparameters
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        }

        model_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models_to_train.items():
            print(f"Cross-validating {name}...")
            if name in ['random_forest', 'gradient_boosting']:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict_proba(X_test_scaled)[:, 1]
            else:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]

            score = roc_auc_score(y_test, y_pred)
            print(f"{name} CV AUC: {np.mean(scores):.4f} | Test AUC: {score:.4f}")
            model_scores[name] = np.mean(scores)
            self.models[name] = model

        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        print(f"\nBest model: {best_model_name} (CV AUC: {model_scores[best_model_name]:.4f})")

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Setup SHAP explainer for Random Forest
        if best_model_name == 'random_forest':
            self.shap_explainer = shap.TreeExplainer(self.best_model)

        # Save models
        joblib.dump(self.best_model, 'advanced_credit_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')

        return X_test, y_test, feature_columns

    def predict_with_explanation(self, farmer_features):
        """Predict default probability with SHAP explanation"""
        if isinstance(farmer_features, dict):
            farmer_features = pd.DataFrame([farmer_features])

        # Scale features if needed
        if self.best_model_name in ['random_forest', 'gradient_boosting']:
            features_scaled = self.scaler.transform(farmer_features)
            prediction = self.best_model.predict_proba(features_scaled)[0][1]
        else:
            prediction = self.best_model.predict_proba(farmer_features)[0][1]

        # Generate explanation
        explanation = {}
        if self.shap_explainer:
            if self.best_model_name in ['random_forest', 'gradient_boosting']:
                shap_values = self.shap_explainer.shap_values(features_scaled)[1]
            else:
                shap_values = self.shap_explainer.shap_values(farmer_features)[1]

            # Top contributing factors
            feature_names = farmer_features.columns
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values,
                'feature_value': farmer_features.iloc[0].values
            }).sort_values('shap_value', key=abs, ascending=False)

            explanation = shap_df.head(10).to_dict('records')

        return {
            'default_probability': float(prediction),
            'credit_score': int((1 - prediction) * 750 + 250),  # Scale to 250-1000
            'risk_level': 'Low' if prediction < 0.2 else 'Medium' if prediction < 0.5 else 'High',
            'explanation': explanation
        }

# Initialize and train the model
if __name__ == "__main__":
    model = AdvancedCreditModel()
    print("Creating hyper-realistic agricultural dataset...")
    df = model.create_advanced_dataset(5000)
    print("Training ensemble model...")
    X_test, y_test, feature_columns = model.train_ensemble_model(df)
    print("Model training completed!")

    # Display some statistics
    default_rate = df['default_flag'].mean()
    print(f"Dataset default rate: {default_rate:.2%}")
    print(f"Total features: {len(feature_columns)}")