# CARBON CREDIT AGENT
# Intelligent agent that tracks sustainable practices and issues tokenized carbon credits

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import numpy as np
import hashlib

from agentic_core import BaseAgent, AgentAction, PerceptionData, simulate_api_call, calculate_confidence_score, generate_blockchain_hash

class CarbonCreditAgent(BaseAgent):
    """
    AI Agent that manages carbon credit lifecycle
    - Tracks sustainable farming practices
    - Calculates carbon sequestration potential
    - Issues tokenized carbon credits with blockchain hashes
    - Matches credits with buyers/investors
    """
    
    def __init__(self):
        super().__init__(
            name="CarbonCreditAgent",
            description="Carbon sequestration tracking and credit tokenization"
        )
        self.carbon_practices = {
            'cover_crops': {'sequestration_rate': 0.5, 'verification_difficulty': 'medium'},
            'no_till_farming': {'sequestration_rate': 0.8, 'verification_difficulty': 'easy'},
            'agroforestry': {'sequestration_rate': 2.1, 'verification_difficulty': 'hard'},
            'organic_farming': {'sequestration_rate': 0.3, 'verification_difficulty': 'medium'},
            'crop_rotation': {'sequestration_rate': 0.4, 'verification_difficulty': 'easy'},
            'composting': {'sequestration_rate': 0.2, 'verification_difficulty': 'easy'},
            'biochar_application': {'sequestration_rate': 1.5, 'verification_difficulty': 'hard'}
        }
        
        self.carbon_prices = {
            'voluntary_market': {'price_per_ton': 15.0, 'currency': 'USD'},
            'compliance_market': {'price_per_ton': 25.0, 'currency': 'USD'},
            'premium_verified': {'price_per_ton': 35.0, 'currency': 'USD'}
        }
        
    async def perceive(self, context: Dict[str, Any]) -> List[PerceptionData]:
        """Collect data about farming practices and carbon potential"""
        perceptions = []
        reasoning_trace = ["🌱 PERCEPTION: Analyzing carbon sequestration potential..."]
        
        # Get farmer profile and practices
        farmer_data = context.get('farmer_profile', {})
        perceptions.append(PerceptionData(
            source="farmer_profile",
            data_type="farming_practices",
            content=farmer_data,
            timestamp=datetime.now(),
            reliability_score=0.95
        ))
        reasoning_trace.append(f"✓ Farmer: {farmer_data.get('name', 'Unknown')}, Land: {farmer_data.get('land_size', 0)}ha")
        
        # Simulate satellite/IoT data for practice verification
        satellite_data = {
            'ndvi_trend': random.uniform(0.3, 0.8),  # Vegetation index
            'soil_cover_percentage': random.uniform(60, 95),
            'tillage_detected': random.choice([True, False]),
            'tree_cover_change': random.uniform(-5, 15),  # % change in tree cover
            'biomass_estimate': random.uniform(2.5, 8.5)  # tons/hectare
        }
        
        perceptions.append(PerceptionData(
            source="satellite_monitoring",
            data_type="practice_verification", 
            content=satellite_data,
            timestamp=datetime.now(),
            reliability_score=0.88
        ))
        reasoning_trace.append(f"🛰️ Satellite verification: NDVI={satellite_data['ndvi_trend']:.2f}, Soil cover={satellite_data['soil_cover_percentage']:.0f}%")
        
        # Get soil health data for carbon baseline
        soil_data = simulate_api_call("soil_data", {
            'lat': farmer_data.get('latitude', 28.6),
            'lon': farmer_data.get('longitude', 77.2)
        })
        perceptions.append(PerceptionData(
            source="soil_sensors",
            data_type="carbon_baseline",
            content=soil_data,
            timestamp=datetime.now(),
            reliability_score=0.82
        ))
        reasoning_trace.append(f"🌍 Soil analysis: Organic matter={soil_data.get('organic_matter', 0):.1f}%, pH={soil_data.get('ph_level', 7):.1f}")
        
        # Historical practice data (simulated IoT/farmer reports)
        practice_history = {
            'practices_implemented': random.sample(list(self.carbon_practices.keys()), k=random.randint(2, 5)),
            'implementation_months': random.randint(6, 36),
            'practice_consistency': random.uniform(0.7, 0.95),
            'verification_documents': random.randint(3, 8)
        }
        
        perceptions.append(PerceptionData(
            source="practice_tracking",
            data_type="historical_practices",
            content=practice_history,
            timestamp=datetime.now(),
            reliability_score=0.91
        ))
        reasoning_trace.append(f"📋 Practice history: {len(practice_history['practices_implemented'])} practices over {practice_history['implementation_months']} months")
        
        self.current_reasoning_trace = reasoning_trace
        return perceptions
        
    async def reason(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """Calculate carbon sequestration and credit potential"""
        reasoning_trace = self.current_reasoning_trace + ["", "🧮 REASONING: Calculating carbon credits..."]
        
        # Extract data from perceptions
        farmer_data = {}
        satellite_data = {}
        soil_data = {}
        practice_history = {}
        
        for perception in perceptions:
            if perception.source == "farmer_profile":
                farmer_data = perception.content
            elif perception.source == "satellite_monitoring":
                satellite_data = perception.content
            elif perception.source == "soil_sensors":
                soil_data = perception.content
            elif perception.source == "practice_tracking":
                practice_history = perception.content
                
        # Calculate base carbon sequestration
        land_size = farmer_data.get('land_size', 2.0)
        base_sequestration = land_size * 1.2  # Base 1.2 tons CO2/hectare/year
        
        reasoning_trace.append(f"📏 Base sequestration: {base_sequestration:.2f} tons CO2/year ({land_size:.1f}ha × 1.2)")
        
        # Calculate practice-based additions
        practice_bonus = 0
        verified_practices = []
        
        for practice in practice_history.get('practices_implemented', []):
            if practice in self.carbon_practices:
                practice_info = self.carbon_practices[practice]
                sequestration_bonus = land_size * practice_info['sequestration_rate']
                
                # Apply verification confidence
                verification_confidence = self._get_verification_confidence(
                    practice, satellite_data, practice_history
                )
                
                verified_sequestration = sequestration_bonus * verification_confidence
                practice_bonus += verified_sequestration
                
                verified_practices.append({
                    'practice': practice,
                    'theoretical_bonus': sequestration_bonus,
                    'verified_bonus': verified_sequestration,
                    'confidence': verification_confidence
                })
                
                reasoning_trace.append(f"🌾 {practice}: +{verified_sequestration:.2f} tons CO2 (confidence: {verification_confidence:.1%})")
        
        total_sequestration = base_sequestration + practice_bonus
        reasoning_trace.append(f"📊 Total sequestration: {total_sequestration:.2f} tons CO2/year")
        
        # Determine credit eligibility and market
        credit_eligibility = self._assess_credit_eligibility(
            total_sequestration, verified_practices, practice_history, satellite_data
        )
        
        # Calculate credit value
        market_type = self._determine_market_type(credit_eligibility['quality_score'])
        credit_value = self._calculate_credit_value(total_sequestration, market_type)
        
        reasoning_trace.append(f"🏆 Credit quality: {credit_eligibility['quality_tier']} ({credit_eligibility['quality_score']:.1%})")
        reasoning_trace.append(f"💰 Market type: {market_type}, Value: ${credit_value:.0f}")
        
        reasoning = {
            'farmer_profile': farmer_data,
            'total_sequestration': total_sequestration,
            'verified_practices': verified_practices,
            'credit_eligibility': credit_eligibility,
            'market_type': market_type,
            'credit_value': credit_value,
            'satellite_verification': satellite_data,
            'reasoning_trace': reasoning_trace
        }
        
        return reasoning
        
    async def act(self, reasoning: Dict[str, Any]) -> AgentAction:
        """Issue carbon credits and create blockchain tokens"""
        reasoning_trace = reasoning['reasoning_trace'] + ["", "🪙 ACTION: Tokenizing carbon credits..."]
        
        # Generate unique carbon credit certificate
        credit_certificate = {
            'credit_id': f"CARBON_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(10000, 99999)}",
            'farmer_id': reasoning['farmer_profile'].get('farmer_id', 'unknown'),
            'farmer_name': reasoning['farmer_profile'].get('name', 'Unknown Farmer'),
            'issue_date': datetime.now().isoformat(),
            'vintage_year': datetime.now().year,
            'sequestration_amount': round(reasoning['total_sequestration'], 3),
            'credit_type': 'Agricultural Soil Carbon',
            'methodology': 'VM0021 - Improved Agricultural Land Management',
            'verification_standard': 'VCS (Verified Carbon Standard)',
            'quality_tier': reasoning['credit_eligibility']['quality_tier'],
            'practices_verified': [p['practice'] for p in reasoning['verified_practices']],
            'geographic_coordinates': {
                'lat': reasoning['farmer_profile'].get('latitude', 28.6),
                'lon': reasoning['farmer_profile'].get('longitude', 77.2)
            },
            'blockchain_hash': generate_blockchain_hash(f"carbon_credit_{reasoning['farmer_profile'].get('farmer_id', 'unknown')}"),
            'estimated_value_usd': reasoning['credit_value'],
            'market_readiness': reasoning['credit_eligibility']['market_ready']
        }
        
        # Create buyer recommendations
        buyer_matches = self._find_potential_buyers(credit_certificate, reasoning['market_type'])
        
        # Generate trading recommendations
        trading_strategy = {
            'recommended_action': self._get_trading_recommendation(reasoning['market_type'], reasoning['credit_value']),
            'price_range': {
                'minimum': reasoning['credit_value'] * 0.85,
                'target': reasoning['credit_value'],
                'premium': reasoning['credit_value'] * 1.15
            },
            'optimal_timing': self._get_optimal_timing(),
            'buyer_matches': buyer_matches
        }
        
        # Calculate confidence score
        confidence_factors = {
            'sequestration_accuracy': reasoning['credit_eligibility']['quality_score'],
            'practice_verification': np.mean([p['confidence'] for p in reasoning['verified_practices']]) if reasoning['verified_practices'] else 0.5,
            'satellite_reliability': 0.88,
            'market_demand': random.uniform(0.7, 0.9)
        }
        confidence_score = calculate_confidence_score(confidence_factors)
        
        reasoning_trace.append(f"🆔 Credit ID: {credit_certificate['credit_id']}")
        reasoning_trace.append(f"🔐 Blockchain hash: {credit_certificate['blockchain_hash']}")
        reasoning_trace.append(f"💎 Quality tier: {credit_certificate['quality_tier']}")
        reasoning_trace.append(f"🎯 Recommended buyers: {len(buyer_matches)}")
        reasoning_trace.append(f"📈 Confidence: {confidence_score:.2%}")
        
        action = AgentAction(
            agent_name=self.name,
            action_type="CARBON_CREDIT_ISSUANCE",
            inputs={
                'farmer_profile': reasoning['farmer_profile'],
                'sequestration_data': reasoning['verified_practices']
            },
            outputs={
                'credit_certificate': credit_certificate,
                'trading_strategy': trading_strategy,
                'summary': f"Issued {reasoning['total_sequestration']:.2f} tons CO2 credits worth ${reasoning['credit_value']:.0f}"
            },
            reasoning_trace=reasoning_trace,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            execution_time_ms=0
        )
        
        return action
        
    def _get_verification_confidence(self, practice: str, satellite_data: Dict, 
                                   practice_history: Dict) -> float:
        """Calculate confidence level for practice verification"""
        practice_info = self.carbon_practices[practice]
        base_confidence = 0.6
        
        # Satellite data boosts confidence
        if practice == 'no_till_farming' and not satellite_data.get('tillage_detected', True):
            base_confidence += 0.3
        elif practice == 'agroforestry' and satellite_data.get('tree_cover_change', 0) > 5:
            base_confidence += 0.25
        elif practice == 'cover_crops' and satellite_data.get('soil_cover_percentage', 0) > 80:
            base_confidence += 0.2
            
        # Practice consistency matters
        consistency = practice_history.get('practice_consistency', 0.8)
        base_confidence *= consistency
        
        # Documentation verification
        doc_bonus = min(0.1, practice_history.get('verification_documents', 0) * 0.02)
        base_confidence += doc_bonus
        
        return min(0.95, base_confidence)
        
    def _assess_credit_eligibility(self, total_sequestration: float, verified_practices: List,
                                 practice_history: Dict, satellite_data: Dict) -> Dict[str, Any]:
        """Assess overall credit quality and eligibility"""
        
        # Quality scoring factors
        sequestration_score = min(1.0, total_sequestration / 10.0)  # Scale based on 10 tons max
        practice_diversity = len(verified_practices) / 7.0  # Max 7 practices
        verification_avg = np.mean([p['confidence'] for p in verified_practices]) if verified_practices else 0.5
        consistency_score = practice_history.get('practice_consistency', 0.8)
        
        quality_score = (sequestration_score * 0.3 + practice_diversity * 0.2 + 
                        verification_avg * 0.3 + consistency_score * 0.2)
        
        # Determine quality tier
        if quality_score > 0.8:
            quality_tier = "Premium Verified"
        elif quality_score > 0.6:
            quality_tier = "Standard Verified"
        else:
            quality_tier = "Basic Verified"
            
        return {
            'quality_score': quality_score,
            'quality_tier': quality_tier,
            'market_ready': quality_score > 0.5,
            'additionality_verified': True,  # Assume passed additionality test
            'permanence_risk': 'Low' if consistency_score > 0.8 else 'Medium'
        }
        
    def _determine_market_type(self, quality_score: float) -> str:
        """Determine which carbon market is appropriate"""
        if quality_score > 0.8:
            return "premium_verified"
        elif quality_score > 0.6:
            return "compliance_market"
        else:
            return "voluntary_market"
            
    def _calculate_credit_value(self, sequestration: float, market_type: str) -> float:
        """Calculate total value of carbon credits"""
        price_per_ton = self.carbon_prices[market_type]['price_per_ton']
        return sequestration * price_per_ton * 83  # Convert USD to INR
        
    def _find_potential_buyers(self, credit_cert: Dict, market_type: str) -> List[Dict]:
        """Find potential buyers for carbon credits"""
        buyers = [
            {
                'buyer_type': 'Corporate ESG',
                'name': 'TechCorp Sustainability Fund',
                'demand': random.uniform(50, 200),
                'price_willingness': self.carbon_prices[market_type]['price_per_ton'] * random.uniform(0.95, 1.1),
                'match_score': random.uniform(0.7, 0.9)
            },
            {
                'buyer_type': 'Impact Investor',
                'name': 'Green Agriculture Fund',
                'demand': random.uniform(100, 500),
                'price_willingness': self.carbon_prices[market_type]['price_per_ton'] * random.uniform(0.9, 1.05),
                'match_score': random.uniform(0.8, 0.95)
            },
            {
                'buyer_type': 'Financial Institution',
                'name': 'Capital One Green Finance',
                'demand': random.uniform(200, 1000),
                'price_willingness': self.carbon_prices[market_type]['price_per_ton'] * random.uniform(1.0, 1.2),
                'match_score': random.uniform(0.85, 0.98)
            }
        ]
        
        # Filter buyers based on credit amount and quality
        credit_amount = credit_cert['sequestration_amount']
        return [b for b in buyers if b['demand'] >= credit_amount and b['match_score'] > 0.7]
        
    def _get_trading_recommendation(self, market_type: str, credit_value: float) -> str:
        """Get trading strategy recommendation"""
        if market_type == "premium_verified":
            return "HOLD - Wait for premium buyers, market trending upward"
        elif credit_value > 50000:  # High value credits
            return "BATCH_SELL - Bundle with other credits for bulk pricing"
        else:
            return "IMMEDIATE_SELL - Good market conditions, sell now"
            
    def _get_optimal_timing(self) -> Dict[str, str]:
        """Provide timing recommendations"""
        return {
            'best_month': 'December',  # End of year corporate buying
            'avoid_month': 'July',     # Mid-year budget constraints  
            'market_cycle': 'Bullish', # Current market sentiment
            'urgency': 'Medium'        # How quickly to act
        }