"""
Explainable AI Core Module for AgriCredAI
Provides data provenance, confidence scores, and human-interpretable explanations
Implements SHAP explainability and comprehensive reasoning traces
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import hashlib
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataProvenance:
    """Tracks the origin and quality of data used in AI decisions"""
    source_name: str
    source_type: str  # "api", "database", "user_input", "fallback", "simulation"
    data_url: Optional[str] = None
    api_key_used: bool = False
    last_updated: Optional[datetime] = None
    data_freshness: str = "unknown"  # "real_time", "hourly", "daily", "weekly", "monthly", "unknown"
    coverage_area: str = "unknown"  # "point", "regional", "national", "global", "unknown"
    verification_status: str = "unknown"  # "verified", "estimated", "unverified"
    confidence_score: float = 0.0
    fallback_reason: Optional[str] = None
    data_quality_score: float = 0.0
    sample_size: Optional[int] = None
    error_margin: Optional[float] = None

@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scores"""
    overall_confidence: float
    data_quality_confidence: float
    model_confidence: float
    feature_confidence: float
    temporal_confidence: float
    spatial_confidence: float
    factors: Dict[str, float]
    limitations: List[str]
    uncertainty_sources: List[str]

@dataclass
class AIExplanation:
    """Human-interpretable explanation of AI decision"""
    decision_summary: str
    key_factors: List[Dict[str, Any]]
    reasoning_trace: List[str]
    alternative_scenarios: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    confidence_explanation: str
    limitations_disclaimer: str
    data_citations: List[DataProvenance]
    last_updated: datetime

@dataclass
class ExplainableOutput:
    """Complete explainable AI output"""
    output_id: str
    timestamp: datetime
    primary_output: Any
    confidence_score: float
    confidence_breakdown: ConfidenceBreakdown
    explanation: AIExplanation
    data_provenance: List[DataProvenance]
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[np.ndarray] = None
    model_metadata: Optional[Dict[str, Any]] = None

class ExplainableAICore:
    """Core explainable AI functionality"""
    
    def __init__(self):
        self.explanation_templates = self._load_explanation_templates()
        self.confidence_calibrators = {}
        self.feature_descriptions = self._load_feature_descriptions()
        
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load templates for different types of explanations"""
        return {
            "credit_approval": "Based on {key_factors}, we recommend {decision} with {confidence}% confidence. The main factors are: {factor_list}",
            "weather_alert": "Weather alert for {location}: {condition} with {confidence}% confidence. Based on data from {data_sources}",
            "market_prediction": "Market prediction for {commodity}: {prediction} with {confidence}% confidence. Factors include: {factor_list}",
            "soil_recommendation": "Soil recommendation for {crop}: {recommendation} with {confidence}% confidence. Based on {soil_data}",
            "risk_assessment": "Risk assessment: {risk_level} risk with {confidence}% confidence. Key risk factors: {risk_factors}"
        }
    
    def _load_feature_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Load human-readable descriptions of ML features"""
        return {
            "farmer_age": {
                "description": "Farmer's age in years",
                "impact": "Younger farmers may have higher technology adoption but less experience",
                "unit": "years"
            },
            "land_size": {
                "description": "Total cultivated land area",
                "impact": "Larger farms may have economies of scale but higher operational complexity",
                "unit": "hectares"
            },
            "education_level": {
                "description": "Farmer's education level (1-5 scale)",
                "impact": "Higher education correlates with better financial management and technology adoption",
                "unit": "scale 1-5"
            },
            "irrigation_access": {
                "description": "Access to irrigation facilities",
                "impact": "Irrigation reduces weather dependency and improves crop yield stability",
                "unit": "binary (0/1)"
            },
            "payment_history_score": {
                "description": "Historical loan repayment performance",
                "impact": "Past behavior is a strong predictor of future repayment",
                "unit": "score 0-1"
            },
            "debt_to_income_ratio": {
                "description": "Ratio of debt payments to monthly income",
                "impact": "Lower ratios indicate better debt management capacity",
                "unit": "ratio"
            },
            "soil_health_index": {
                "description": "Overall soil health and fertility score",
                "impact": "Better soil health leads to higher crop yields and lower input costs",
                "unit": "score 0-1"
            },
            "weather_risk": {
                "description": "Weather-related risk factors",
                "impact": "Weather risks can significantly affect crop yields and repayment capacity",
                "unit": "score 0-1"
            },
            "market_volatility": {
                "description": "Market price volatility for crops",
                "impact": "High volatility increases income uncertainty and repayment risk",
                "unit": "coefficient of variation"
            }
        }
    
    def calculate_confidence_score(self, 
                                 data_quality: float,
                                 model_performance: float,
                                 feature_completeness: float,
                                 temporal_relevance: float,
                                 spatial_coverage: float) -> ConfidenceBreakdown:
        """Calculate comprehensive confidence score with breakdown"""
        
        # Weighted combination of confidence factors
        weights = {
            'data_quality': 0.25,
            'model_performance': 0.30,
            'feature_completeness': 0.20,
            'temporal_relevance': 0.15,
            'spatial_coverage': 0.10
        }
        
        overall_confidence = (
            data_quality * weights['data_quality'] +
            model_performance * weights['model_performance'] +
            feature_completeness * weights['feature_completeness'] +
            temporal_relevance * weights['temporal_relevance'] +
            spatial_coverage * weights['spatial_coverage']
        )
        
        # Identify factors affecting confidence
        factors = {
            'data_quality': data_quality,
            'model_performance': model_performance,
            'feature_completeness': feature_completeness,
            'temporal_relevance': temporal_relevance,
            'spatial_coverage': spatial_coverage
        }
        
        # Identify limitations
        limitations = []
        if data_quality < 0.7:
            limitations.append("Limited data quality may affect accuracy")
        if model_performance < 0.8:
            limitations.append("Model performance below optimal threshold")
        if feature_completeness < 0.9:
            limitations.append("Some important features are missing")
        if temporal_relevance < 0.8:
            limitations.append("Data may not reflect current conditions")
        if spatial_coverage < 0.7:
            limitations.append("Limited geographic coverage")
        
        # Identify uncertainty sources
        uncertainty_sources = []
        if overall_confidence < 0.6:
            uncertainty_sources.append("Multiple factors contributing to uncertainty")
        if data_quality < 0.6:
            uncertainty_sources.append("Low quality input data")
        if model_performance < 0.7:
            uncertainty_sources.append("Suboptimal model performance")
        
        return ConfidenceBreakdown(
            overall_confidence=overall_confidence,
            data_quality_confidence=data_quality,
            model_confidence=model_performance,
            feature_confidence=feature_completeness,
            temporal_confidence=temporal_relevance,
            spatial_confidence=spatial_coverage,
            factors=factors,
            limitations=limitations,
            uncertainty_sources=uncertainty_sources
        )
    
    def generate_credit_explanation(self, 
                                  farmer_data: Dict[str, Any],
                                  prediction: float,
                                  confidence: float,
                                  model: Any,
                                  feature_names: List[str]) -> AIExplanation:
        """Generate human-readable explanation for credit decisions"""
        
        # Extract key factors
        key_factors = []
        for feature in feature_names:
            if feature in farmer_data and feature in self.feature_descriptions:
                value = farmer_data[feature]
                desc = self.feature_descriptions[feature]
                
                # Determine impact direction
                if feature in ['payment_history_score', 'education_level', 'irrigation_access', 'soil_health_index']:
                    impact = "positive" if value > 0.5 else "negative"
                elif feature in ['debt_to_income_ratio', 'weather_risk', 'market_volatility']:
                    impact = "negative" if value > 0.5 else "positive"
                else:
                    impact = "neutral"
                
                key_factors.append({
                    'feature': feature,
                    'value': value,
                    'description': desc['description'],
                    'impact': impact,
                    'unit': desc['unit'],
                    'reasoning': desc['impact']
                })
        
        # Sort by importance (you can use SHAP values here)
        key_factors.sort(key=lambda x: abs(x['value']), reverse=True)
        
        # Generate decision summary
        if prediction < 0.3:
            decision = "APPROVE"
            risk_level = "Low"
        elif prediction < 0.6:
            decision = "REVIEW"
            risk_level = "Medium"
        else:
            decision = "REJECT"
            risk_level = "High"
        
        # Generate reasoning trace
        reasoning_trace = [
            f"Analyzed {len(feature_names)} factors for credit assessment",
            f"Primary risk factors identified: {', '.join([f['feature'] for f in key_factors[:3]])}",
            f"Overall risk assessment: {risk_level} risk",
            f"Recommendation: {decision} with {confidence:.1%} confidence"
        ]
        
        # Generate alternative scenarios
        alternative_scenarios = [
            {
                "scenario": "Conservative approach",
                "threshold": 0.4,
                "outcome": "More stringent approval criteria",
                "impact": "Lower default rate, reduced loan volume"
            },
            {
                "scenario": "Aggressive approach", 
                "threshold": 0.7,
                "outcome": "More lenient approval criteria",
                "impact": "Higher loan volume, increased default risk"
            }
        ]
        
        # Risk assessment
        risk_assessment = {
            "overall_risk": risk_level,
            "risk_score": prediction,
            "key_risk_factors": [f for f in key_factors if f['impact'] == 'negative'][:3],
            "mitigation_strategies": self._generate_risk_mitigation_strategies(key_factors)
        }
        
        # Recommendations
        recommendations = []
        if decision == "REJECT":
            recommendations.extend([
                "Focus on improving payment history",
                "Consider reducing debt burden",
                "Explore government subsidy schemes"
            ])
        elif decision == "REVIEW":
            recommendations.extend([
                "Provide additional documentation",
                "Consider co-signer or collateral",
                "Start with smaller loan amount"
            ])
        else:
            recommendations.extend([
                "Maintain current financial practices",
                "Consider expanding operations",
                "Explore additional financial products"
            ])
        
        return AIExplanation(
            decision_summary=f"Credit {decision} - {risk_level} risk with {confidence:.1%} confidence",
            key_factors=key_factors,
            reasoning_trace=reasoning_trace,
            alternative_scenarios=alternative_scenarios,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            confidence_explanation=f"Confidence is based on data quality ({confidence:.1%}), model performance, and feature completeness",
            limitations_disclaimer="This assessment is based on available data and may not capture all relevant factors",
            data_citations=[],  # Will be populated by caller
            last_updated=datetime.now()
        )
    
    def generate_weather_explanation(self,
                                   weather_data: Dict[str, Any],
                                   location: str,
                                   crop_type: str,
                                   confidence: float) -> AIExplanation:
        """Generate explanation for weather-based recommendations"""
        
        # Extract weather conditions
        temp = weather_data.get('main', {}).get('temp', 25)
        humidity = weather_data.get('main', {}).get('humidity', 60)
        description = weather_data.get('weather', [{}])[0].get('description', 'unknown')
        
        # Determine weather impact on crops
        weather_impact = self._assess_weather_crop_impact(temp, humidity, description, crop_type)
        
        # Key factors
        key_factors = [
            {
                'feature': 'temperature',
                'value': temp,
                'description': 'Current temperature',
                'impact': 'optimal' if 20 <= temp <= 35 else 'suboptimal',
                'unit': '°C',
                'reasoning': f"Temperature {temp}°C is {'optimal' if 20 <= temp <= 35 else 'suboptimal'} for {crop_type}"
            },
            {
                'feature': 'humidity',
                'value': humidity,
                'description': 'Current humidity level',
                'impact': 'optimal' if 40 <= humidity <= 80 else 'suboptimal',
                'unit': '%',
                'reasoning': f"Humidity {humidity}% is {'optimal' if 40 <= humidity <= 80 else 'suboptimal'} for {crop_type}"
            },
            {
                'feature': 'weather_condition',
                'value': description,
                'description': 'Current weather condition',
                'impact': weather_impact['overall'],
                'unit': 'description',
                'reasoning': weather_impact['reasoning']
            }
        ]
        
        # Reasoning trace
        reasoning_trace = [
            f"Analyzed weather conditions for {location}",
            f"Temperature: {temp}°C ({weather_impact['temp_impact']})",
            f"Humidity: {humidity}% ({weather_impact['humidity_impact']})",
            f"Weather: {description} ({weather_impact['condition_impact']})",
            f"Overall assessment: {weather_impact['overall']} conditions for {crop_type}"
        ]
        
        # Alternative scenarios
        alternative_scenarios = [
            {
                "scenario": "Temperature increase by 5°C",
                "outcome": "Potential heat stress",
                "impact": "Reduced crop yield, increased water requirement"
            },
            {
                "scenario": "Humidity increase by 20%",
                "outcome": "Potential disease risk",
                "impact": "Fungal diseases, reduced air circulation"
            }
        ]
        
        # Risk assessment
        risk_assessment = {
            "weather_risk": weather_impact['risk_level'],
            "crop_vulnerability": weather_impact['crop_vulnerability'],
            "mitigation_needed": weather_impact['mitigation_needed'],
            "recommended_actions": weather_impact['recommended_actions']
        }
        
        # Recommendations
        recommendations = weather_impact['recommended_actions']
        
        return AIExplanation(
            decision_summary=f"Weather conditions in {location}: {weather_impact['overall']} for {crop_type}",
            key_factors=key_factors,
            reasoning_trace=reasoning_trace,
            alternative_scenarios=alternative_scenarios,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            confidence_explanation=f"Confidence based on weather data quality and location accuracy",
            limitations_disclaimer="Weather conditions can change rapidly; monitor local forecasts",
            data_citations=[],  # Will be populated by caller
            last_updated=datetime.now()
        )
    
    def _assess_weather_crop_impact(self, temp: float, humidity: float, description: str, crop_type: str) -> Dict[str, Any]:
        """Assess impact of weather conditions on specific crops"""
        
        # Crop-specific optimal conditions
        crop_conditions = {
            'rice': {'temp_range': (20, 35), 'humidity_range': (60, 90), 'water_loving': True},
            'wheat': {'temp_range': (15, 25), 'humidity_range': (40, 70), 'water_loving': False},
            'cotton': {'temp_range': (20, 35), 'humidity_range': (50, 80), 'water_loving': False},
            'sugarcane': {'temp_range': (20, 35), 'humidity_range': (60, 85), 'water_loving': True}
        }
        
        crop_specs = crop_conditions.get(crop_type.lower(), crop_conditions['wheat'])
        
        # Temperature impact
        if temp < crop_specs['temp_range'][0]:
            temp_impact = "too cold"
            temp_risk = "high"
        elif temp > crop_specs['temp_range'][1]:
            temp_impact = "too hot"
            temp_risk = "high"
        else:
            temp_impact = "optimal"
            temp_risk = "low"
        
        # Humidity impact
        if humidity < crop_specs['humidity_range'][0]:
            humidity_impact = "too dry"
            humidity_risk = "medium"
        elif humidity > crop_specs['humidity_range'][1]:
            humidity_impact = "too humid"
            humidity_risk = "medium"
        else:
            humidity_impact = "optimal"
            humidity_risk = "low"
        
        # Weather condition impact
        if 'rain' in description.lower():
            if crop_specs['water_loving']:
                condition_impact = "beneficial"
                condition_risk = "low"
            else:
                condition_impact = "moderate risk"
                condition_risk = "medium"
        elif 'storm' in description.lower() or 'thunder' in description.lower():
            condition_impact = "high risk"
            condition_risk = "high"
        else:
            condition_impact = "stable"
            condition_risk = "low"
        
        # Overall assessment
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}
        overall_risk = max(risk_levels[temp_risk], risk_levels[humidity_risk], risk_levels[condition_risk])
        
        if overall_risk == 1:
            overall = "favorable"
            risk_level = "low"
        elif overall_risk == 2:
            overall = "moderate"
            risk_level = "medium"
        else:
            overall = "unfavorable"
            risk_level = "high"
        
        # Recommended actions
        recommended_actions = []
        if temp_risk == "high":
            if temp < crop_specs['temp_range'][0]:
                recommended_actions.append("Consider frost protection measures")
            else:
                recommended_actions.append("Implement heat stress mitigation")
        
        if humidity_risk == "high":
            if humidity < crop_specs['humidity_range'][0]:
                recommended_actions.append("Increase irrigation frequency")
            else:
                recommended_actions.append("Improve field drainage and ventilation")
        
        if condition_risk == "high":
            recommended_actions.append("Monitor weather alerts and take protective measures")
        
        return {
            'overall': overall,
            'risk_level': risk_level,
            'temp_impact': temp_impact,
            'humidity_impact': humidity_impact,
            'condition_impact': condition_impact,
            'crop_vulnerability': 'high' if overall_risk >= 2 else 'low',
            'mitigation_needed': len(recommended_actions) > 0,
            'recommended_actions': recommended_actions,
            'reasoning': f"Current conditions are {overall} for {crop_type} cultivation"
        }
    
    def _generate_risk_mitigation_strategies(self, key_factors: List[Dict[str, Any]]) -> List[str]:
        """Generate risk mitigation strategies based on key factors"""
        strategies = []
        
        for factor in key_factors:
            if factor['impact'] == 'negative':
                if factor['feature'] == 'debt_to_income_ratio':
                    strategies.append("Focus on debt reduction and income diversification")
                elif factor['feature'] == 'weather_risk':
                    strategies.append("Implement weather insurance and protective measures")
                elif factor['feature'] == 'market_volatility':
                    strategies.append("Consider forward contracts and price hedging")
                elif factor['feature'] == 'soil_health_index':
                    strategies.append("Implement soil improvement and conservation practices")
        
        return strategies
    
    def create_shap_explanation(self, 
                               model: Any,
                               input_data: np.ndarray,
                               feature_names: List[str]) -> Dict[str, Any]:
        """Create SHAP-based explanation for model predictions"""
        try:
            if hasattr(model, 'predict_proba'):
                # Classification model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                # For binary classification, use the positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
            else:
                # Regression model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
            
            # Get feature importance
            feature_importance = {}
            for i, feature in enumerate(feature_names):
                feature_importance[feature] = float(np.abs(shap_values[0, i]))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create explanation
            explanation = {
                'shap_values': shap_values.tolist(),
                'feature_importance': dict(sorted_features),
                'top_features': [f[0] for f in sorted_features[:5]],
                'explanation_text': f"Top 5 most important features: {', '.join([f[0] for f in sorted_features[:5]])}"
            }
            
            return explanation
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {
                'shap_values': None,
                'feature_importance': {},
                'top_features': [],
                'explanation_text': "Feature importance analysis not available"
            }
    
    def create_explainable_output(self,
                                output_id: str,
                                primary_output: Any,
                                confidence_breakdown: ConfidenceBreakdown,
                                explanation: AIExplanation,
                                data_provenance: List[DataProvenance],
                                feature_importance: Optional[Dict[str, float]] = None,
                                shap_values: Optional[np.ndarray] = None,
                                model_metadata: Optional[Dict[str, Any]] = None) -> ExplainableOutput:
        """Create a complete explainable AI output"""
        
        return ExplainableOutput(
            output_id=output_id,
            timestamp=datetime.now(),
            primary_output=primary_output,
            confidence_score=confidence_breakdown.overall_confidence,
            confidence_breakdown=confidence_breakdown,
            explanation=explanation,
            data_provenance=data_provenance,
            feature_importance=feature_importance,
            shap_values=shap_values.tolist() if shap_values is not None else None,
            model_metadata=model_metadata
        )
    
    def export_explanation(self, explainable_output: ExplainableOutput, format: str = 'json') -> str:
        """Export explanation in various formats"""
        if format == 'json':
            return json.dumps(asdict(explainable_output), default=str, indent=2)
        elif format == 'text':
            return self._format_explanation_as_text(explainable_output)
        elif format == 'html':
            return self._format_explanation_as_html(explainable_output)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_explanation_as_text(self, output: ExplainableOutput) -> str:
        """Format explanation as plain text"""
        text = f"""
EXPLAINABLE AI OUTPUT
====================
Output ID: {output.output_id}
Timestamp: {output.timestamp}
Confidence: {output.confidence_score:.1%}

DECISION SUMMARY
===============
{output.explanation.decision_summary}

KEY FACTORS
==========
"""
        for factor in output.explanation.key_factors[:5]:
            text += f"• {factor['feature']}: {factor['value']} ({factor['unit']}) - {factor['impact']}\n"
        
        text += f"""
REASONING
=========
"""
        for step in output.explanation.reasoning_trace:
            text += f"• {step}\n"
        
        text += f"""
RECOMMENDATIONS
==============
"""
        for rec in output.explanation.recommendations:
            text += f"• {rec}\n"
        
        text += f"""
CONFIDENCE BREAKDOWN
===================
Overall: {output.confidence_breakdown.overall_confidence:.1%}
Data Quality: {output.confidence_breakdown.data_quality_confidence:.1%}
Model Performance: {output.confidence_breakdown.model_confidence:.1%}

LIMITATIONS
==========
"""
        for limitation in output.confidence_breakdown.limitations:
            text += f"• {limitation}\n"
        
        return text
    
    def _format_explanation_as_html(self, output: ExplainableOutput) -> str:
        """Format explanation as HTML"""
        html = f"""
        <div class="explainable-output">
            <h2>AI Decision Explanation</h2>
            <div class="confidence-score">
                <strong>Confidence:</strong> {output.confidence_score:.1%}
            </div>
            <div class="decision-summary">
                <h3>Decision Summary</h3>
                <p>{output.explanation.decision_summary}</p>
            </div>
            <div class="key-factors">
                <h3>Key Factors</h3>
                <ul>
        """
        
        for factor in output.explanation.key_factors[:5]:
            html += f"""
                    <li>
                        <strong>{factor['feature']}:</strong> {factor['value']} ({factor['unit']})
                        <br><em>Impact:</em> {factor['impact']}
                        <br><em>Reasoning:</em> {factor['reasoning']}
                    </li>
            """
        
        html += """
                </ul>
            </div>
        </div>
        """
        
        return html

# Global instance
explainable_ai = ExplainableAICore()
