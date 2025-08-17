# MARKET ADVISORY AGENT  
# Intelligent agent that provides market intelligence and trading recommendations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import numpy as np

from agentic_core import BaseAgent, AgentAction, PerceptionData, simulate_api_call, calculate_confidence_score

class MarketAdvisoryAgent(BaseAgent):
    """
    AI Agent that provides intelligent market advisory services
    - Analyzes market trends and price patterns
    - Provides sell/hold/buy recommendations  
    - Predicts optimal timing for commodity trading
    - Considers weather and supply chain factors
    """
    
    def __init__(self):
        super().__init__(
            name="MarketAdvisoryAgent", 
            description="Real-time market analysis and trading recommendations"
        )
        
        self.crop_seasonality = {
            'wheat': {'harvest_months': [4, 5], 'peak_price_months': [8, 9, 10]},
            'rice': {'harvest_months': [10, 11], 'peak_price_months': [6, 7, 8]},
            'cotton': {'harvest_months': [10, 11, 12], 'peak_price_months': [3, 4, 5]},
            'sugarcane': {'harvest_months': [1, 2, 3], 'peak_price_months': [8, 9]},
            'soybean': {'harvest_months': [10, 11], 'peak_price_months': [5, 6, 7]},
            'maize': {'harvest_months': [7, 8], 'peak_price_months': [2, 3, 4]}
        }
        
        self.market_indicators = [
            'mandi_prices', 'export_demand', 'storage_levels', 'transport_costs',
            'weather_impact', 'government_policy', 'international_prices', 'currency_rates'
        ]
        
    async def perceive(self, context: Dict[str, Any]) -> List[PerceptionData]:
        """Collect comprehensive market data"""
        perceptions = []
        reasoning_trace = ["📊 PERCEPTION: Gathering market intelligence..."]
        
        # Get farmer and crop information
        farmer_data = context.get('farmer_profile', {})
        crop_type = farmer_data.get('crop_type', 'wheat').lower()
        region = farmer_data.get('region', 'UP')
        
        reasoning_trace.append(f"🌾 Target crop: {crop_type.title()}, Region: {region}")
        
        perceptions.append(PerceptionData(
            source="farmer_profile",
            data_type="crop_info",
            content=farmer_data,
            timestamp=datetime.now(),
            reliability_score=0.98
        ))
        
        # Fetch current market prices
        current_market = simulate_api_call("market_prices", {
            'crop': crop_type,
            'region': region
        })
        
        # Enhance with realistic market data
        current_market.update({
            'historical_avg': current_market['current_price'] * random.uniform(0.85, 1.15),
            'seasonal_high': current_market['current_price'] * random.uniform(1.1, 1.4),
            'seasonal_low': current_market['current_price'] * random.uniform(0.6, 0.9),
            'storage_premium': random.uniform(50, 200),
            'quality_premium': random.uniform(0, 150)
        })
        
        perceptions.append(PerceptionData(
            source="agmarknet",
            data_type="current_prices",
            content=current_market,
            timestamp=datetime.now(),
            reliability_score=0.92
        ))
        reasoning_trace.append(f"💰 Current price: ₹{current_market['current_price']:.0f}/quintal ({current_market['price_trend']})")
        
        # Get weather impact analysis
        weather_data = simulate_api_call("weather", {
            'lat': farmer_data.get('latitude', 28.6),
            'lon': farmer_data.get('longitude', 77.2)
        })
        
        # Add weather-market correlation
        weather_impact = {
            'supply_risk': weather_data.get('drought_risk', 0.3),
            'quality_risk': weather_data.get('rainfall_forecast', 20) / 100,
            'harvest_delay_risk': random.uniform(0.1, 0.5),
            'regional_impact_score': random.uniform(0.2, 0.8)
        }
        weather_data.update(weather_impact)
        
        perceptions.append(PerceptionData(
            source="weather_service",
            data_type="supply_impact",
            content=weather_data,
            timestamp=datetime.now(),
            reliability_score=0.85
        ))
        reasoning_trace.append(f"🌦️ Weather impact: Supply risk={weather_impact['supply_risk']:.2f}, Quality risk={weather_impact['quality_risk']:.2f}")
        
        # Storage and logistics data
        logistics_data = {
            'current_storage_capacity': random.uniform(60, 95),  # % utilization
            'transport_availability': random.uniform(0.7, 0.95),
            'fuel_price_impact': random.uniform(-5, 15),  # % change
            'nearest_mandi_distance': farmer_data.get('nearest_mandi_distance', random.uniform(5, 30)),
            'storage_cost_per_month': random.uniform(15, 35),  # per quintal
            'quality_degradation_rate': random.uniform(0.5, 2.0)  # % per month
        }
        
        perceptions.append(PerceptionData(
            source="logistics_network",
            data_type="supply_chain",
            content=logistics_data,
            timestamp=datetime.now(),
            reliability_score=0.88
        ))
        reasoning_trace.append(f"🚛 Logistics: Storage {logistics_data['current_storage_capacity']:.0f}% full, Transport availability {logistics_data['transport_availability']:.1%}")
        
        # Policy and macro factors
        policy_data = {
            'msp_rate': current_market['current_price'] * random.uniform(0.8, 1.1),
            'export_policy': random.choice(['open', 'restricted', 'duty_imposed']),
            'import_tariff': random.uniform(10, 40),
            'buffer_stock_policy': random.choice(['active_buying', 'stable', 'releasing']),
            'credit_policy_impact': random.uniform(-0.1, 0.2)
        }
        
        perceptions.append(PerceptionData(
            source="policy_monitor",
            data_type="macro_factors",
            content=policy_data,
            timestamp=datetime.now(),
            reliability_score=0.79
        ))
        reasoning_trace.append(f"🏛️ Policy: MSP=₹{policy_data['msp_rate']:.0f}, Export={policy_data['export_policy']}")
        
        self.current_reasoning_trace = reasoning_trace
        return perceptions
        
    async def reason(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """Analyze market data and generate trading insights"""
        reasoning_trace = self.current_reasoning_trace + ["", "🔍 REASONING: Analyzing market opportunities..."]
        
        # Extract data from perceptions
        farmer_data = {}
        market_data = {}
        weather_data = {}
        logistics_data = {}
        policy_data = {}
        
        for perception in perceptions:
            if perception.source == "farmer_profile":
                farmer_data = perception.content
            elif perception.source == "agmarknet":
                market_data = perception.content
            elif perception.source == "weather_service":
                weather_data = perception.content
            elif perception.source == "logistics_network":
                logistics_data = perception.content
            elif perception.source == "policy_monitor":
                policy_data = perception.content
        
        crop_type = farmer_data.get('crop_type', 'wheat').lower()
        current_month = datetime.now().month
        
        # Price trend analysis
        current_price = market_data.get('current_price', 2000)
        historical_avg = market_data.get('historical_avg', current_price)
        price_vs_avg = (current_price - historical_avg) / historical_avg
        
        reasoning_trace.append(f"📈 Price analysis: Current vs historical = {price_vs_avg:+.1%}")
        
        # Seasonal analysis
        seasonality = self.crop_seasonality.get(crop_type, self.crop_seasonality['wheat'])
        is_harvest_season = current_month in seasonality['harvest_months']
        is_peak_price_season = current_month in seasonality['peak_price_months']
        
        seasonal_factor = "Harvest pressure" if is_harvest_season else "Peak demand" if is_peak_price_season else "Neutral"
        reasoning_trace.append(f"📅 Seasonal factor: {seasonal_factor}")
        
        # Supply-demand analysis
        supply_pressure = self._calculate_supply_pressure(weather_data, logistics_data, is_harvest_season)
        demand_strength = self._calculate_demand_strength(market_data, policy_data, weather_data)
        
        reasoning_trace.append(f"⚖️ Supply pressure: {supply_pressure:.2f}, Demand strength: {demand_strength:.2f}")
        
        # Price prediction
        price_forecast = self._forecast_prices(
            current_price, supply_pressure, demand_strength, seasonality, current_month
        )
        
        # Risk assessment
        price_risk = self._assess_price_risks(weather_data, policy_data, logistics_data)
        
        # Generate recommendation
        recommendation = self._generate_trading_recommendation(
            price_vs_avg, supply_pressure, demand_strength, price_forecast, 
            price_risk, is_harvest_season, is_peak_price_season
        )
        
        reasoning_trace.append(f"🎯 Primary recommendation: {recommendation['action']}")
        reasoning_trace.append(f"💡 Expected price movement: {price_forecast['direction']} ({price_forecast['magnitude']:+.1%})")
        
        reasoning = {
            'farmer_profile': farmer_data,
            'market_analysis': {
                'current_price': current_price,
                'price_vs_historical': price_vs_avg,
                'supply_pressure': supply_pressure,
                'demand_strength': demand_strength,
                'seasonal_factor': seasonal_factor
            },
            'price_forecast': price_forecast,
            'risk_assessment': price_risk,
            'recommendation': recommendation,
            'reasoning_trace': reasoning_trace
        }
        
        return reasoning
        
    async def act(self, reasoning: Dict[str, Any]) -> AgentAction:
        """Generate actionable market advisory"""
        reasoning_trace = reasoning['reasoning_trace'] + ["", "💼 ACTION: Generating market advisory..."]
        
        # Create comprehensive market advisory
        market_advisory = {
            'advisory_id': f"MKT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            'farmer_id': reasoning['farmer_profile'].get('farmer_id', 'unknown'),
            'crop_type': reasoning['farmer_profile'].get('crop_type', 'Unknown'),
            'issue_date': datetime.now().isoformat(),
            'validity_days': 7,
            
            'primary_recommendation': reasoning['recommendation'],
            'price_forecast': reasoning['price_forecast'],
            'risk_factors': reasoning['risk_assessment']['high_risk_factors'],
            
            'action_plan': self._create_action_plan(reasoning),
            'timing_guidance': self._get_timing_guidance(reasoning),
            'profit_projections': self._calculate_profit_projections(reasoning),
            
            'market_insights': [
                f"Current price is {reasoning['market_analysis']['price_vs_historical']:+.1%} vs historical average",
                f"Supply pressure: {'High' if reasoning['market_analysis']['supply_pressure'] > 0.6 else 'Moderate' if reasoning['market_analysis']['supply_pressure'] > 0.3 else 'Low'}",
                f"Demand outlook: {'Strong' if reasoning['market_analysis']['demand_strength'] > 0.6 else 'Moderate' if reasoning['market_analysis']['demand_strength'] > 0.3 else 'Weak'}",
                f"Seasonal factor: {reasoning['market_analysis']['seasonal_factor']}"
            ],
            
            'alternative_strategies': self._get_alternative_strategies(reasoning)
        }
        
        # Calculate confidence score
        confidence_factors = {
            'data_quality': 0.88,  # Average of perception reliabilities
            'forecast_accuracy': 1 - reasoning['risk_assessment']['overall_risk'],
            'market_stability': 1 - reasoning['price_forecast']['volatility'],
            'seasonal_clarity': 0.9 if reasoning['market_analysis']['seasonal_factor'] != 'Neutral' else 0.7
        }
        confidence_score = calculate_confidence_score(confidence_factors)
        
        reasoning_trace.append(f"📋 Advisory ID: {market_advisory['advisory_id']}")
        reasoning_trace.append(f"🎯 Action: {market_advisory['primary_recommendation']['action']}")
        reasoning_trace.append(f"📊 Confidence: {confidence_score:.1%}")
        reasoning_trace.append(f"💰 Profit potential: {market_advisory['profit_projections']['expected_return']:+.1%}")
        
        action = AgentAction(
            agent_name=self.name,
            action_type="MARKET_ADVISORY",
            inputs={
                'farmer_profile': reasoning['farmer_profile'],
                'market_conditions': reasoning['market_analysis']
            },
            outputs={
                'market_advisory': market_advisory,
                'summary': f"{market_advisory['primary_recommendation']['action']} - Expected {market_advisory['price_forecast']['direction']} price movement"
            },
            reasoning_trace=reasoning_trace,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            execution_time_ms=0
        )
        
        return action
        
    def _calculate_supply_pressure(self, weather_data: Dict, logistics_data: Dict, 
                                 is_harvest_season: bool) -> float:
        """Calculate supply pressure index (0-1, higher = more supply pressure)"""
        base_pressure = 0.5
        
        # Weather-related supply impacts
        if weather_data.get('supply_risk', 0) > 0.5:
            base_pressure -= 0.2  # Reduced supply due to weather
        
        # Harvest season increases supply
        if is_harvest_season:
            base_pressure += 0.3
            
        # Storage capacity impacts
        storage_util = logistics_data.get('current_storage_capacity', 80) / 100
        if storage_util > 0.9:
            base_pressure += 0.2  # Need to sell due to storage constraints
            
        return np.clip(base_pressure, 0, 1)
        
    def _calculate_demand_strength(self, market_data: Dict, policy_data: Dict, 
                                 weather_data: Dict) -> float:
        """Calculate demand strength index (0-1, higher = stronger demand)"""
        base_demand = 0.5
        
        # Price trend indicates demand
        if market_data.get('price_trend') == 'rising':
            base_demand += 0.2
        elif market_data.get('price_trend') == 'falling':
            base_demand -= 0.2
            
        # Export policy impacts
        if policy_data.get('export_policy') == 'open':
            base_demand += 0.15
        elif policy_data.get('export_policy') == 'restricted':
            base_demand -= 0.1
            
        # Weather impacts on other regions (creates demand)
        regional_impact = weather_data.get('regional_impact_score', 0.5)
        if regional_impact > 0.7:
            base_demand += 0.15  # Other regions need supply
            
        return np.clip(base_demand, 0, 1)
        
    def _forecast_prices(self, current_price: float, supply_pressure: float, 
                        demand_strength: float, seasonality: Dict, current_month: int) -> Dict:
        """Generate price forecast"""
        
        # Base forecast from supply-demand balance
        balance = demand_strength - supply_pressure
        expected_change = balance * 0.15  # Max 15% change from balance
        
        # Seasonal adjustment
        peak_months = seasonality['peak_price_months']
        if current_month in peak_months:
            seasonal_boost = 0.08
        elif (current_month + 1) % 12 + 1 in peak_months:
            seasonal_boost = 0.05  # Approaching peak season
        else:
            seasonal_boost = 0
            
        total_change = expected_change + seasonal_boost
        
        # Volatility estimate
        volatility = abs(supply_pressure - 0.5) + abs(demand_strength - 0.5)
        
        direction = "Bullish" if total_change > 0.03 else "Bearish" if total_change < -0.03 else "Stable"
        
        return {
            'direction': direction,
            'magnitude': total_change,
            'target_price': current_price * (1 + total_change),
            'volatility': volatility,
            'confidence': 1 - volatility,
            'timeline_days': random.randint(15, 45)
        }
        
    def _assess_price_risks(self, weather_data: Dict, policy_data: Dict, 
                           logistics_data: Dict) -> Dict:
        """Assess various price risks"""
        risks = []
        risk_scores = []
        
        # Weather risks
        supply_risk = weather_data.get('supply_risk', 0.3)
        if supply_risk > 0.6:
            risks.append("High weather-related supply disruption risk")
            risk_scores.append(supply_risk)
            
        # Policy risks
        if policy_data.get('export_policy') == 'restricted':
            risks.append("Export restrictions limiting demand")
            risk_scores.append(0.4)
            
        # Logistics risks
        transport_avail = logistics_data.get('transport_availability', 0.8)
        if transport_avail < 0.7:
            risks.append("Limited transport availability")
            risk_scores.append(1 - transport_avail)
            
        overall_risk = np.mean(risk_scores) if risk_scores else 0.2
        
        return {
            'overall_risk': overall_risk,
            'high_risk_factors': risks,
            'risk_level': 'High' if overall_risk > 0.6 else 'Medium' if overall_risk > 0.3 else 'Low'
        }
        
    def _generate_trading_recommendation(self, price_vs_avg: float, supply_pressure: float,
                                       demand_strength: float, price_forecast: Dict, 
                                       price_risk: Dict, is_harvest: bool, is_peak: bool) -> Dict:
        """Generate primary trading recommendation"""
        
        # Decision logic
        if price_vs_avg > 0.1 and price_forecast['direction'] == 'Bearish':
            action = "SELL_NOW"
            reason = "Prices above average but forecast declining"
            urgency = "High"
        elif price_forecast['direction'] == 'Bullish' and not is_harvest:
            action = "HOLD_FOR_PEAK"
            reason = "Bullish forecast suggests higher prices ahead"
            urgency = "Low"
        elif is_harvest and price_vs_avg < -0.05:
            action = "STRATEGIC_HOLD"  
            reason = "Harvest pressure depressing prices, wait for seasonal recovery"
            urgency = "Medium"
        elif demand_strength > 0.7:
            action = "SELL_GRADUALLY"
            reason = "Strong demand supports current pricing"
            urgency = "Medium"
        else:
            action = "MONITOR_MARKET"
            reason = "Mixed signals, await clearer trend"
            urgency = "Low"
            
        return {
            'action': action,
            'reason': reason,
            'urgency': urgency,
            'optimal_percentage': self._get_sell_percentage(action),
            'timeframe': self._get_action_timeframe(action, urgency)
        }
        
    def _create_action_plan(self, reasoning: Dict) -> List[Dict]:
        """Create step-by-step action plan"""
        recommendation = reasoning['recommendation']
        
        if recommendation['action'] == "SELL_NOW":
            return [
                {"step": 1, "action": "Prepare harvest for immediate sale", "timeline": "1-2 days"},
                {"step": 2, "action": "Contact multiple buyers for best price", "timeline": "2-3 days"}, 
                {"step": 3, "action": "Complete sale transaction", "timeline": "3-5 days"}
            ]
        elif recommendation['action'] == "HOLD_FOR_PEAK":
            return [
                {"step": 1, "action": "Arrange proper storage facility", "timeline": "1 week"},
                {"step": 2, "action": "Monitor daily price movements", "timeline": "Ongoing"},
                {"step": 3, "action": "Sell when target price reached", "timeline": "2-6 weeks"}
            ]
        else:
            return [
                {"step": 1, "action": "Continue monitoring market conditions", "timeline": "Daily"},
                {"step": 2, "action": "Prepare for flexible selling strategy", "timeline": "1 week"},
                {"step": 3, "action": "Execute based on market signals", "timeline": "As needed"}
            ]
            
    def _get_timing_guidance(self, reasoning: Dict) -> Dict:
        """Provide timing guidance"""
        forecast = reasoning['price_forecast']
        
        return {
            'best_selling_window': f"Next {forecast['timeline_days']} days",
            'avoid_selling_during': "Major festivals or market holidays",
            'monitor_closely': "Weather forecasts and government policy announcements",
            'review_strategy_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        }
        
    def _calculate_profit_projections(self, reasoning: Dict) -> Dict:
        """Calculate profit projections for different strategies"""
        current_price = reasoning['market_analysis']['current_price']
        forecast = reasoning['price_forecast']
        
        immediate_sale = 0  # Baseline
        strategic_hold = forecast['magnitude']  # Expected price change
        
        return {
            'immediate_sale_return': immediate_sale,
            'strategic_hold_return': strategic_hold,
            'expected_return': strategic_hold if reasoning['recommendation']['action'] == "HOLD_FOR_PEAK" else immediate_sale,
            'risk_adjusted_return': strategic_hold * (1 - reasoning['risk_assessment']['overall_risk'])
        }
        
    def _get_alternative_strategies(self, reasoning: Dict) -> List[Dict]:
        """Provide alternative trading strategies"""
        return [
            {
                "strategy": "Gradual Selling",
                "description": "Sell 30% now, 40% in 2 weeks, 30% at peak",
                "pros": "Reduces risk, captures average price",
                "cons": "May miss optimal selling point"
            },
            {
                "strategy": "Contract Farming",
                "description": "Pre-sell to buyers at fixed price",
                "pros": "Price certainty, guaranteed buyer",
                "cons": "May miss upside potential"
            },
            {
                "strategy": "Value Addition",
                "description": "Process crop for higher margins",
                "pros": "Higher profit potential",
                "cons": "Requires additional investment and time"
            }
        ]
        
    def _get_sell_percentage(self, action: str) -> int:
        """Get recommended selling percentage"""
        percentages = {
            "SELL_NOW": 80,
            "SELL_GRADUALLY": 40,
            "STRATEGIC_HOLD": 20,
            "HOLD_FOR_PEAK": 0,
            "MONITOR_MARKET": 0
        }
        return percentages.get(action, 30)
        
    def _get_action_timeframe(self, action: str, urgency: str) -> str:
        """Get timeframe for action"""
        if urgency == "High":
            return "1-3 days"
        elif urgency == "Medium":
            return "1-2 weeks"
        else:
            return "2-6 weeks"