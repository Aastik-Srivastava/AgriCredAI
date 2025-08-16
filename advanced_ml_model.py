import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold

# The AdvancedCreditModel class is designed to simulate, train, and explain advanced credit risk models for agricultural lending using a rich set of features. It begins by initializing key components such as a dictionary to store trained models, a feature scaler (StandardScaler), and placeholders for feature importance and SHAP explainers, which are used for model interpretability.

# The create_advanced_dataset method generates a synthetic dataset with over 50 features for a specified number of samples (default 5000). These features cover a wide range of domains relevant to agricultural credit risk, including farmer demographics, weather and climate risks, market and economic indicators, financial history, infrastructure, agricultural practices, government support, social factors, supply chain metrics, and risk management behaviors. Each feature is generated using appropriate probability distributions to mimic real-world variability. The method then computes a composite risk score for each sample based on a weighted sum of several risk factors, adding some random noise for realism. The top 25% of samples by risk score are labeled as defaults, creating a binary target variable (default_flag). The method returns the dataset as a pandas DataFrame.

# The train_ensemble_model method takes this dataset and trains an ensemble of machine learning models to predict default risk. It first separates features from the target, splits the data into training and test sets, and scales the features. Four models are trained: Random Forest, XGBoost, LightGBM, and Gradient Boosting, each with tuned hyperparameters. The method evaluates each model using the ROC AUC score on the test set, selects the best-performing model, and stores it as the primary model for future predictions. If the best model supports feature importance, this information is extracted and stored. For interpretability, a SHAP explainer is set up if the best model is a Random Forest. The trained model and scaler are saved to disk for reuse.

# The predict_with_explanation method allows for making predictions on new farmer data and provides interpretability using SHAP values. It accepts input as either a dictionary or DataFrame, scales the features if necessary, and predicts the probability of default using the best model. If a SHAP explainer is available, it computes SHAP values to identify the top contributing features to the prediction, returning these as an explanation. The method outputs the predicted default probability, a derived credit score (scaled between 150 and 1000), a risk level (Low, Medium, or High), and the explanation of the most influential features.

# Overall, this class demonstrates a robust approach to credit risk modeling, combining realistic data simulation, ensemble machine learning, and model interpretability, making it suitable for both experimentation and practical applications in agricultural finance.


class AdvancedCreditModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.shap_explainer = None
        
    def create_advanced_dataset(self, n_samples=5000):
        """Create comprehensive dataset with 50+ features"""
        np.random.seed(42)
        
        data = {}
        
        # Basic farmer information
        data['farmer_id'] = range(n_samples)
        data['land_size'] = np.random.gamma(2, 1.5, n_samples)  # Realistic land distribution
        data['crop_type_encoded'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples)
        data['farmer_age'] = np.random.normal(45, 12, n_samples)
        data['education_level'] = np.random.choice([1, 2, 3, 4, 5], n_samples)  # 1=Illiterate, 5=Graduate
        data['family_size'] = np.random.poisson(4, n_samples)
        
        # Weather and climate features
        data['current_temperature'] = np.random.normal(28, 8, n_samples)
        data['current_humidity'] = np.random.normal(65, 15, n_samples)
        data['temperature_stress'] = np.random.beta(2, 5, n_samples)
        data['humidity_stress'] = np.random.beta(2, 5, n_samples)
        data['frost_risk_7days'] = np.random.beta(1, 10, n_samples)
        data['drought_risk_7days'] = np.random.beta(2, 8, n_samples)
        data['excess_rain_risk'] = np.random.beta(1, 9, n_samples)
        data['seasonal_rainfall_deviation'] = np.random.normal(0, 25, n_samples)
        data['historical_drought_frequency'] = np.random.poisson(1, n_samples)
        data['climate_change_vulnerability'] = np.random.beta(3, 7, n_samples)
        
        # Market and economic features
        data['current_price'] = np.random.gamma(5, 500, n_samples)
        data['price_volatility'] = np.random.beta(2, 8, n_samples)
        data['price_trend'] = np.random.normal(0, 0.15, n_samples)
        data['market_demand_index'] = np.random.beta(4, 6, n_samples)
        data['export_potential'] = np.random.beta(3, 7, n_samples)
        data['storage_price_premium'] = np.random.beta(2, 8, n_samples)
        
        # Financial history and credit features
        data['payment_history_score'] = np.random.beta(4, 2, n_samples)
        data['yield_consistency'] = np.random.beta(5, 3, n_samples)
        data['loan_to_land_ratio'] = np.random.beta(3, 7, n_samples)
        data['debt_to_income_ratio'] = np.random.beta(3, 5, n_samples)
        data['savings_to_income_ratio'] = np.random.beta(2, 8, n_samples)
        data['credit_utilization'] = np.random.beta(3, 7, n_samples)
        data['number_of_credit_sources'] = np.random.poisson(2, n_samples)
        data['informal_lending_dependency'] = np.random.beta(2, 6, n_samples)
        
        # Geographic and infrastructure features
        data['nearest_mandi_distance'] = np.random.gamma(2, 8, n_samples)
        data['irrigation_access'] = np.random.binomial(n=1, p=0.6, size=n_samples) 
        data['connectivity_index'] = np.random.beta(4, 6, n_samples)
        data['road_quality_index'] = np.random.beta(3, 7, n_samples)
        data['electricity_reliability'] = np.random.beta(4, 6, n_samples)
        data['mobile_network_strength'] = np.random.beta(5, 5, n_samples)
        data['bank_branch_distance'] = np.random.gamma(2, 5, n_samples)
        
        # Agricultural practices and technology
        data['mechanization_level'] = np.random.beta(3, 7, n_samples)
        data['seed_quality_index'] = np.random.beta(4, 6, n_samples)
        data['fertilizer_usage_efficiency'] = np.random.beta(4, 6, n_samples)
        data['pest_management_score'] = np.random.beta(4, 6, n_samples)
        data['soil_health_index'] = np.random.beta(5, 5, n_samples)
        data['nutrient_deficiency_risk'] = np.random.beta(2, 8, n_samples)
        data['organic_farming_adoption'] = np.random.beta(2, 8, n_samples)
        data['precision_agriculture_usage'] = np.random.beta(1, 9, n_samples)
        
        # Government schemes and support
        data['eligible_schemes_count'] = np.random.poisson(3, n_samples)
        data['insurance_coverage'] = np.random.binomial(n=1, p=0.4, size=n_samples) 
        data['subsidy_utilization'] = np.random.beta(3, 7, n_samples)
        data['msp_eligibility'] = np.random.binomial(n=1, p=0.7, size=n_samples) 
        data['kisan_credit_card'] = np.random.binomial(n=1, p=0.5, size=n_samples) 
        data['government_training_participation'] = np.random.beta(2, 8, n_samples)
        
        # Social and community features
        data['cooperative_membership'] = np.random.binomial(n=1, p=0.45, size=n_samples) 
        data['community_leadership_role'] = np.random.binomial(n=1, p=0.15, size=n_samples) 
        data['social_capital_index'] = np.random.beta(4, 6, n_samples)
        data['extension_service_access'] = np.random.beta(3, 7, n_samples)
        data['peer_learning_participation'] = np.random.beta(3, 7, n_samples)
        
        # Input cost and supply chain features
        data['input_cost_index'] = np.random.beta(4, 6, n_samples)
        data['labor_availability'] = np.random.beta(4, 6, n_samples)
        data['storage_access'] = np.random.binomial(n=1, p=0.3, size=n_samples) 
        data['transport_cost_burden'] = np.random.beta(3, 7, n_samples)
        data['supply_chain_integration'] = np.random.beta(2, 8, n_samples)
        
        # Risk management features
        data['diversification_index'] = np.random.beta(3, 7, n_samples)
        data['technology_adoption'] = np.random.beta(3, 7, n_samples)
        data['disaster_preparedness'] = np.random.beta(2, 8, n_samples)
        data['alternative_income_sources'] = np.random.beta(3, 7, n_samples)
        data['livestock_ownership'] = np.random.binomial(n=1, p=0.6, size=n_samples) 
        
        # Create realistic default probability
        # Complex interaction of factors
        risk_score = (
            0.1 * (1 - data['payment_history_score']) +
            0.08 * data['drought_risk_7days'] +
            0.07 * data['frost_risk_7days'] +
            0.06 * data['price_volatility'] +
            0.06 * data['loan_to_land_ratio'] +
            0.05 * (1 - data['yield_consistency']) +
            0.05 * data['debt_to_income_ratio'] +
            0.04 * (data['nearest_mandi_distance'] / 50) +
            0.04 * (1 - data['soil_health_index']) +
            0.04 * data['temperature_stress'] +
            0.03 * (1 - data['irrigation_access']) +
            0.03 * data['nutrient_deficiency_risk'] +
            0.03 * (1 - data['connectivity_index']) +
            0.02 * data['excess_rain_risk'] +
            0.02 * (1 - data['insurance_coverage']) +
            0.02 * data['input_cost_index'] +
            0.02 * (1 - data['diversification_index']) +
            0.01 * (1 - data['cooperative_membership']) +
            0.01 * data['informal_lending_dependency'] +
            0.01 * (1 - data['technology_adoption']) +
            np.random.normal(0, 0.05, n_samples)  # Random noise
        )
        
        # Convert to binary default flag
        data['default_flag'] = (risk_score > np.percentile(risk_score, 75)).astype(int)
        
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

        # Add tuned logistic regression
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
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

        # Select best model by CV score
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

        # Setup SHAP explainer
        if best_model_name == 'random_forest':
            self.shap_explainer = shap.TreeExplainer(self.best_model)

        # Save everything
        joblib.dump(self.best_model, 'advanced_credit_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')

        return X_test, y_test, feature_columns
    
    
    def predict_with_explanation(self, farmer_features):
        """Predict default probability with SHAP explanation"""
        # Convert to DataFrame if needed
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
            'credit_score': int((1 - prediction) * 850 + 150),
            'risk_level': 'Low' if prediction < 0.3 else 'Medium' if prediction < 0.6 else 'High',
            'explanation': explanation
        }

# Initialize and train the model
if __name__ == "__main__":
    model = AdvancedCreditModel()
    
    print("Creating advanced dataset...")
    df = model.create_advanced_dataset(5000)
    
    print("Training ensemble model...")
    X_test, y_test, feature_columns = model.train_ensemble_model(df)
    
    print("Model training completed!")
