# DYNAMIC FINANCING AGENT
# Intelligent agent that adapts loan structures based on crop cycles, weather risks, and farmer profiles

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import numpy as np

from agentic_core import BaseAgent, AgentAction, PerceptionData, simulate_api_call, calculate_confidence_score

class DynamicFinancingAgent(BaseAgent):
    """
    AI Agent that provides intelligent financing recommendations
    - Analyzes farmer risk profiles
    - Suggests optimal loan structures  
    - Adapts repayment schedules based on crop cycles and weather
    - Learns from financing outcomes
    """
    
    def __init__(self):
        super().__init__(
            name="DynamicFinancingAgent",
            description="Intelligent loan structuring and risk-adaptive financing"
        )
        self.loan_products = {
            'crop_cycle_loan': {
                'rate_range': (8.5, 12.5),
                'duration_months': [6, 9, 12],
                'collateral_required': 0.7
            },
            'equipment_financing': {
                'rate_range': (10.0, 15.0),  
                'duration_months': [24, 36, 48],
                'collateral_required': 0.8
            },
            'working_capital': {
                'rate_range': (12.0, 18.0),
                'duration_months': [3, 6, 12],
                'collateral_required': 0.5
            },
            'climate_resilient_loan': {
                'rate_range': (7.5, 11.0),
                'duration_months': [12, 18, 24],
                'collateral_required': 0.6
            }
        }
        
    async def perceive(self, context: Dict[str, Any]) -> List[PerceptionData]:
        """Gather data for financing decision"""
        perceptions = []
        
        reasoning_trace = ["🔍 PERCEPTION: Gathering financing-relevant data..."]
        
        # Get farmer profile data
        farmer_data = context.get('farmer_profile', {})
        perceptions.append(PerceptionData(
            source="farmer_profile",
            data_type="demographics",
            content=farmer_data,
            timestamp=datetime.now(),
            reliability_score=0.95
        ))
        reasoning_trace.append(f"✓ Farmer profile: {farmer_data.get('name', 'Unknown')}, Land: {farmer_data.get('land_size', 0)}ha")
        
        # Fetch weather risk data
        weather_data = simulate_api_call("weather", {
            'lat': farmer_data.get('latitude', 28.6),
            'lon': farmer_data.get('longitude', 77.2)
        })
        perceptions.append(PerceptionData(
            source="weather_api",
            data_type="risk_assessment",
            content=weather_data,
            timestamp=datetime.now(),
            reliability_score=0.85
        ))
        reasoning_trace.append(f"✓ Weather risk: Drought={weather_data.get('drought_risk', 0):.2f}, Temp={weather_data.get('temperature', 0):.1f}°C")
        
        # Get market data for crop pricing
        market_data = simulate_api_call("market_prices", {
            'crop': farmer_data.get('crop_type', 'wheat'),
            'region': farmer_data.get('region', 'UP')
        })
        perceptions.append(PerceptionData(
            source="agmarknet",
            data_type="market_prices",
            content=market_data,
            timestamp=datetime.now(),
            reliability_score=0.90
        ))
        reasoning_trace.append(f"✓ Market data: Price={market_data.get('current_price', 0):.0f}/quintal, Trend={market_data.get('price_trend', 'stable')}")
        
        # Get credit history if available
        credit_score = context.get('credit_assessment', {}).get('credit_score', random.randint(450, 750))
        credit_data = {
            'credit_score': credit_score,
            'payment_history': context.get('credit_assessment', {}).get('payment_history_score', random.uniform(0.7, 0.95)),
            'debt_to_income': context.get('credit_assessment', {}).get('debt_to_income_ratio', random.uniform(0.2, 0.8))
        }
        perceptions.append(PerceptionData(
            source="credit_bureau",
            data_type="creditworthiness",
            content=credit_data,
            timestamp=datetime.now(),
            reliability_score=0.92
        ))
        reasoning_trace.append(f"✓ Credit profile: Score={credit_score}, Payment history={credit_data['payment_history']:.2f}")
        
        self.current_reasoning_trace = reasoning_trace
        return perceptions
        
    async def reason(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """Analyze data and determine optimal financing strategy"""
        reasoning_trace = self.current_reasoning_trace + ["", "🧠 REASONING: Analyzing financing options..."]
        
        # Extract data from perceptions
        farmer_data = {}
        weather_data = {}
        market_data = {}
        credit_data = {}
        
        for perception in perceptions:
            if perception.source == "farmer_profile":
                farmer_data = perception.content
            elif perception.source == "weather_api":
                weather_data = perception.content
            elif perception.source == "agmarknet":
                market_data = perception.content
            elif perception.source == "credit_bureau":
                credit_data = perception.content
        
        # Risk assessment
        weather_risk = weather_data.get('drought_risk', 0.3) + weather_data.get('volatility', 0.2)
        market_risk = market_data.get('volatility', 0.2)
        credit_risk = 1 - (credit_data.get('credit_score', 500) / 850)
        
        overall_risk = (weather_risk * 0.4 + market_risk * 0.3 + credit_risk * 0.3)
        reasoning_trace.append(f"📊 Risk Assessment: Weather={weather_risk:.2f}, Market={market_risk:.2f}, Credit={credit_risk:.2f}")
        reasoning_trace.append(f"📈 Overall Risk Score: {overall_risk:.2f}")
        
        # Determine recommended loan product
        land_size = farmer_data.get('land_size', 2.0)
        crop_type = farmer_data.get('crop_type', 'wheat')
        
        # Loan amount calculation based on land size and crop type
        crop_income_multiplier = {
            'wheat': 45000, 'rice': 50000, 'cotton': 80000, 
            'sugarcane': 120000, 'soybean': 55000, 'maize': 40000
        }
        base_income = land_size * crop_income_multiplier.get(crop_type.lower(), 50000)
        max_loan_amount = base_income * 0.6  # 60% of expected annual income
        
        # Select optimal product based on risk and farmer profile
        if overall_risk < 0.3 and weather_data.get('drought_risk', 0) < 0.4:
            recommended_product = 'climate_resilient_loan'
            reasoning_trace.append("🌱 Low risk profile - Recommending Climate Resilient Loan with preferential rates")
        elif land_size > 5 and credit_data.get('credit_score', 500) > 650:
            recommended_product = 'equipment_financing'
            reasoning_trace.append("🚜 Large farm + good credit - Recommending Equipment Financing")
        elif market_data.get('price_trend') == 'rising':
            recommended_product = 'crop_cycle_loan'
            reasoning_trace.append("📈 Rising prices - Recommending Crop Cycle Loan")
        else:
            recommended_product = 'working_capital'
            reasoning_trace.append("💼 Standard profile - Recommending Working Capital loan")
        
        # Calculate interest rate based on risk
        product_info = self.loan_products[recommended_product]
        min_rate, max_rate = product_info['rate_range']
        interest_rate = min_rate + (max_rate - min_rate) * overall_risk
        
        reasoning_trace.append(f"💰 Calculated interest rate: {interest_rate:.2f}% (risk-adjusted)")
        
        # Adaptive repayment schedule
        repayment_schedule = self._create_adaptive_schedule(
            farmer_data, weather_data, market_data, max_loan_amount, recommended_product
        )
        reasoning_trace.append(f"📅 Adaptive schedule: {len(repayment_schedule)} installments")
        
        reasoning = {
            'recommended_product': recommended_product,
            'loan_amount': max_loan_amount,
            'interest_rate': interest_rate,
            'repayment_schedule': repayment_schedule,
            'risk_assessment': {
                'overall_risk': overall_risk,
                'weather_risk': weather_risk,
                'market_risk': market_risk,
                'credit_risk': credit_risk
            },
            'farmer_profile': farmer_data,
            'reasoning_trace': reasoning_trace
        }
        
        return reasoning
        
    async def act(self, reasoning: Dict[str, Any]) -> AgentAction:
        """Generate financing recommendation action"""
        reasoning_trace = reasoning['reasoning_trace'] + ["", "⚡ ACTION: Generating financing package..."]
        
        # Create financing package
        financing_package = {
            'loan_id': f"AGRI_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            'product_type': reasoning['recommended_product'],
            'loan_amount': round(reasoning['loan_amount'], 2),
            'interest_rate': round(reasoning['interest_rate'], 2),
            'tenure_months': len(reasoning['repayment_schedule']),
            'repayment_schedule': reasoning['repayment_schedule'],
            'collateral_requirement': self.loan_products[reasoning['recommended_product']]['collateral_required'],
            'approval_probability': self._calculate_approval_probability(reasoning),
            'risk_category': self._categorize_risk(reasoning['risk_assessment']['overall_risk']),
            'special_features': self._get_special_features(reasoning),
            'next_steps': [
                "Submit loan application with required documents",
                "Property valuation and verification",
                "Final approval and disbursement",
                "Setup automated weather-based payment adjustments"
            ]
        }
        
        # Calculate confidence score
        confidence_factors = {
            'credit_score': 1 - reasoning['risk_assessment']['credit_risk'],
            'market_stability': 1 - reasoning['risk_assessment']['market_risk'],
            'weather_reliability': 1 - reasoning['risk_assessment']['weather_risk'],
            'data_quality': 0.9  # High quality mock data
        }
        confidence_score = calculate_confidence_score(confidence_factors)
        
        reasoning_trace.append(f"✅ Generated loan package: {financing_package['loan_id']}")
        reasoning_trace.append(f"📊 Confidence score: {confidence_score:.2f}")
        reasoning_trace.append(f"🎯 Approval probability: {financing_package['approval_probability']:.1%}")
        
        action = AgentAction(
            agent_name=self.name,
            action_type="FINANCING_RECOMMENDATION",
            inputs={
                'farmer_profile': reasoning['farmer_profile'],
                'risk_assessment': reasoning['risk_assessment']
            },
            outputs={
                'financing_package': financing_package,
                'recommendation_summary': f"Recommended {financing_package['product_type']} of ₹{financing_package['loan_amount']:,.0f} at {financing_package['interest_rate']:.2f}% interest"
            },
            reasoning_trace=reasoning_trace,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            execution_time_ms=0  # Will be set by orchestrator
        )
        
        return action
        
    def _create_adaptive_schedule(self, farmer_data: Dict, weather_data: Dict, 
                                 market_data: Dict, loan_amount: float, product_type: str) -> List[Dict]:
        """Create adaptive repayment schedule based on crop cycles and weather"""
        schedule = []
        duration_months = random.choice(self.loan_products[product_type]['duration_months'])
        monthly_payment = loan_amount / duration_months
        
        # Adjust payments based on crop cycle (harvest months get higher payments)
        crop_type = farmer_data.get('crop_type', 'wheat').lower()
        harvest_months = {
            'wheat': [4, 5],  # April-May
            'rice': [10, 11], # October-November  
            'cotton': [10, 11, 12], # October-December
            'sugarcane': [1, 2, 3], # January-March
            'soybean': [10, 11], # October-November
            'maize': [7, 8] # July-August
        }
        
        current_month = datetime.now().month
        for i in range(duration_months):
            payment_month = (current_month + i) % 12 + 1
            
            # Higher payments during harvest months
            if payment_month in harvest_months.get(crop_type, []):
                payment_multiplier = 1.5
                payment_type = "Harvest Payment"
            else:
                payment_multiplier = 0.7
                payment_type = "Reduced Payment"
                
            # Weather-based adjustments
            if weather_data.get('drought_risk', 0) > 0.6 and payment_multiplier > 1.0:
                payment_multiplier = 1.2  # Reduce even harvest payments if drought risk
                payment_type += " (Weather Adjusted)"
            
            schedule.append({
                'month': i + 1,
                'calendar_month': payment_month,
                'payment_amount': round(monthly_payment * payment_multiplier, 2),
                'payment_type': payment_type,
                'weather_adjustment': weather_data.get('drought_risk', 0) > 0.6
            })
            
        return schedule
        
    def _calculate_approval_probability(self, reasoning: Dict) -> float:
        """Calculate probability of loan approval"""
        risk_score = reasoning['risk_assessment']['overall_risk']
        
        # Higher risk = lower approval probability
        base_approval = 0.95 - (risk_score * 0.6)
        
        # Adjustments for specific factors
        farmer_data = reasoning['farmer_profile']
        if farmer_data.get('land_size', 0) > 3:
            base_approval += 0.05  # Larger farms get bonus
        if farmer_data.get('irrigation_access', 0) == 1:
            base_approval += 0.08  # Irrigation reduces risk
        if farmer_data.get('insurance_coverage', 0) == 1:
            base_approval += 0.06  # Insurance coverage helps
            
        return max(0.1, min(0.98, base_approval))
        
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        if risk_score < 0.25:
            return "Low Risk"
        elif risk_score < 0.5:
            return "Medium Risk"
        else:
            return "High Risk"
            
    def _get_special_features(self, reasoning: Dict) -> List[str]:
        """Get special features based on farmer profile and risk"""
        features = []
        
        farmer_data = reasoning['farmer_profile']
        risk_assessment = reasoning['risk_assessment']
        
        if farmer_data.get('organic_farming', False):
            features.append("🌱 Green Finance Discount: 0.5% rate reduction")
            
        if farmer_data.get('cooperative_membership', False):
            features.append("🤝 Cooperative Member Benefit: Flexible collateral terms")
            
        if risk_assessment['weather_risk'] > 0.5:
            features.append("🌦️ Weather Protection: Automatic payment deferral during extreme weather")
            
        if farmer_data.get('technology_adoption', 0) > 0.7:
            features.append("📱 Digital Farmer Bonus: Mobile payment incentives")
            
        if not features:
            features.append("📋 Standard loan terms apply")
            
        return features