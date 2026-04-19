# AGRI INTELLIGENCE CORE ENGINE
# This module implements the core expert intelligence framework with perception-reasoning-action-feedback cycles

import asyncio
import json
import uuid
from datetime import datetime, timedelta
import random
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib

@dataclass
class IntelAction:
    """Represents an action taken by an intelligence component"""
    agent_name: str
    action_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning_trace: List[str]
    confidence_score: float
    timestamp: datetime
    execution_time_ms: int
    # Added data provenance fields
    data_sources: List[Dict[str, Any]] = None  # List of data sources used in this action
    data_transformations: List[str] = None  # List of transformations applied to the data
    decision_factors: Dict[str, float] = None  # Factors that influenced the decision and their weights
    action_id: str = None  # Unique identifier for this action
    parent_actions: List[str] = None  # IDs of parent actions that led to this action
    # Added explanation fields
    explanation: str = None  # Human-readable explanation of the action
    confidence_breakdown: Dict[str, Any] = None  # Detailed breakdown of confidence score
    alternative_actions: List[Dict[str, Any]] = None  # Alternative actions that were considered
    limitations: List[str] = None  # Known limitations of this action
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.data_transformations is None:
            self.data_transformations = []
        if self.decision_factors is None:
            self.decision_factors = {}
        if self.action_id is None:
            self.action_id = str(uuid.uuid4())
        if self.parent_actions is None:
            self.parent_actions = []
        if self.confidence_breakdown is None:
            self.confidence_breakdown = {}
        if self.alternative_actions is None:
            self.alternative_actions = []
        if self.limitations is None:
            self.limitations = []

@dataclass
class PerceptionData:
    """Raw data collected from various sources"""
    source: str
    data_type: str
    content: Dict[str, Any]
    timestamp: datetime
    reliability_score: float
    # Added data provenance fields
    data_origin: str = "unknown"  # e.g., "api", "database", "user_input", "simulation"
    data_version: str = "1.0"
    collection_method: str = "unknown"  # e.g., "api_call", "database_query", "sensor_reading"
    processing_steps: List[str] = None  # List of processing steps applied to the data
    access_permissions: List[str] = None  # Who can access this data
    retention_policy: str = "standard"  # Data retention policy
    
    def __post_init__(self):
        if self.processing_steps is None:
            self.processing_steps = []
        if self.access_permissions is None:
            self.access_permissions = ["system"]

class BaseAdvisor(ABC):
    """Abstract base class for all expert intelligence components"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.action_history: List[IntelAction] = []
        self.learning_memory: Dict[str, Any] = {}
        
    @abstractmethod
    async def perceive(self, context: Dict[str, Any]) -> List[PerceptionData]:
        """Collect and process input data"""
        pass
        
    @abstractmethod
    async def reason(self, perceptions: List[PerceptionData]) -> Dict[str, Any]:
        """Analyze data and form reasoning"""
        pass
        
    @abstractmethod
    async def act(self, reasoning: Dict[str, Any]) -> IntelAction:
        """Take action based on reasoning"""
        pass
        
    async def feedback(self, action_result: Dict[str, Any]):
        """Learn from action results"""
        # Update learning memory based on feedback
        feedback_key = f"action_{len(self.action_history)}"
        self.learning_memory[feedback_key] = {
            'result': action_result,
            'timestamp': datetime.now(),
            'success_rate': action_result.get('success', False)
        }
        
    def get_reasoning_trace(self) -> List[str]:
        """Return the reasoning process for transparency"""
        if self.action_history:
            return self.action_history[-1].reasoning_trace
        return []

class IntelOrchestrator:
    """Main orchestrator that coordinates multiple expert advisors"""
    
    def __init__(self):
        self.advisors: Dict[str, BaseAdvisor] = {}
        self.global_context: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
    def register_advisor(self, advisor: BaseAdvisor):
        """Register a new advisor with the orchestrator"""
        self.advisors[advisor.name] = advisor
        
    async def run_advisor_cycle(self, advisor_name: str, context: Dict[str, Any]) -> IntelAction:
        """Run a complete perception-reasoning-action cycle for an advisor"""
        start_time = datetime.now()
        
        if advisor_name not in self.advisors:
            raise ValueError(f"Advisor {advisor_name} not found")
            
        advisor = self.advisors[advisor_name]
        
        try:
            # Perception phase
            perceptions = await advisor.perceive(context)
            
            # Reasoning phase  
            reasoning = await advisor.reason(perceptions)
            
            # Action phase
            action = await advisor.act(reasoning)
            
            # Log execution
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            action.execution_time_ms = int(execution_time)
            
            advisor.action_history.append(action)
            
            self.execution_log.append({
                'advisor': advisor_name,
                'action': action,
                'timestamp': datetime.now(),
                'success': True
            })
            
            return action
            
        except Exception as e:
            error_action = IntelAction(
                agent_name=advisor_name,
                action_type="ERROR",
                inputs=context,
                outputs={'error': str(e)},
                reasoning_trace=[f"Error occurred: {str(e)}"],
                confidence_score=0.0,
                timestamp=datetime.now(),
                execution_time_ms=0
            )
            
            self.execution_log.append({
                'agent': agent_name,
                'action': error_action,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
            
            return error_action
    
    async def run_multi_advisor_workflow(self, workflow_context: Dict[str, Any]) -> Dict[str, IntelAction]:
        """Run multiple advisors in coordination"""
        results = {}
        
        # Run advisors in parallel for efficiency
        advisor_tasks = []
        for advisor_name in self.advisors.keys():
            task = self.run_advisor_cycle(advisor_name, workflow_context)
            advisor_tasks.append((advisor_name, task))
        
        # Wait for all advisors to complete
        for advisor_name, task in advisor_tasks:
            try:
                results[advisor_name] = await task
            except Exception as e:
                print(f"Error in advisor {advisor_name}: {e}")
                results[advisor_name] = None
                
        return results
    
    def get_orchestrator_summary(self) -> Dict[str, Any]:
        """Get summary of all advisor activities"""
        summary = {
            'total_advisors': len(self.advisors),
            'total_actions': len(self.execution_log),
            'success_rate': 0,
            'advisors_status': {},
            'recent_actions': self.execution_log[-10:] if self.execution_log else []
        }
        
        if self.execution_log:
            successful_actions = sum(1 for log in self.execution_log if log['success'])
            summary['success_rate'] = successful_actions / len(self.execution_log)
        
        for advisor_name, advisor in self.advisors.items():
            summary['advisors_status'][advisor_name] = {
                'total_actions': len(advisor.action_history),
                'last_action_time': advisor.action_history[-1].timestamp if advisor.action_history else None,
                'learning_entries': len(advisor.learning_memory)
            }
            
        return summary

# Utility functions for agentic operations
def generate_blockchain_hash(data: str) -> str:
    """Generate a mock blockchain-style hash"""
    return hashlib.sha256(f"{data}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

def generate_confidence_explanation(factors: Dict[str, float], weights: Dict[str, float] = None) -> Dict[str, Any]:
    """Generate a detailed explanation of confidence score calculation"""
    if not weights:
        weights = {k: 1.0 for k in factors.keys()}
    
    total_weight = sum(weights.values())
    weighted_factors = {}
    factor_impacts = {}
    
    for k in factors.keys():
        weighted_factors[k] = factors[k] * weights.get(k, 1.0)
        factor_impacts[k] = weighted_factors[k] / total_weight
    
    weighted_sum = sum(weighted_factors.values())
    confidence_score = min(1.0, weighted_sum / total_weight)
    
    # Generate natural language explanations
    factor_explanations = []
    for k, impact in sorted(factor_impacts.items(), key=lambda x: x[1], reverse=True):
        factor_explanations.append({
            'factor': k,
            'raw_value': factors[k],
            'weight': weights.get(k, 1.0),
            'weighted_value': weighted_factors[k],
            'impact_percentage': impact * 100,
            'explanation': f"{k.replace('_', ' ').title()} contributed {impact:.1%} to the confidence score"
        })
    
    return {
        'confidence_score': confidence_score,
        'factors': factor_explanations,
        'calculation_method': 'weighted_average',
        'highest_impact_factor': max(factor_impacts.items(), key=lambda x: x[1])[0] if factor_impacts else None,
        'summary': f"Overall confidence of {confidence_score:.1%} based on {len(factors)} factors"
    }

def simulate_api_call(api_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate API calls for demo purposes"""
    # Add realistic delays
    delay = random.uniform(0.1, 0.5)
    import time
    time.sleep(delay)
    
    # Return mock data based on API type
    if api_name == "weather":
        return {
            'temperature': random.uniform(20, 35),
            'humidity': random.uniform(40, 80),
            'rainfall_forecast': random.uniform(0, 50),
            'drought_risk': random.uniform(0, 1)
        }
    elif api_name == "market_prices":
        return {
            'current_price': random.uniform(1000, 3000),
            'price_trend': random.choice(['rising', 'falling', 'stable']),
            'volatility': random.uniform(0.1, 0.4)
        }
    elif api_name == "soil_data":
        return {
            'ph_level': random.uniform(6.0, 8.0),
            'nitrogen': random.uniform(20, 80),
            'organic_matter': random.uniform(0.5, 3.0),
            'moisture': random.uniform(10, 40)
        }
    else:
        return {'status': 'success', 'data': f'Mock data for {api_name}'}

def track_data_provenance(data_source: str, operation: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a data provenance record for tracking data lineage"""
    if metadata is None:
        metadata = {}
        
    provenance_record = {
        'source': data_source,
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'operation_id': str(uuid.uuid4()),
        'metadata': metadata
    }
    
    return provenance_record

# Export the main classes for use in specific agents
__all__ = ['BaseAdvisor', 'IntelOrchestrator', 'IntelAction', 'PerceptionData', 
           'generate_blockchain_hash', 'calculate_confidence_score', 'simulate_api_call',
           'track_data_provenance', 'generate_confidence_explanation']


def calculate_confidence_score(factors: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate confidence score from multiple factors"""
    explanation = generate_confidence_explanation(factors, weights)
    return explanation['confidence_score']