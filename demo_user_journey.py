"""
Demo and User Journey Module for AgriCredAI
Provides interactive walkthroughs, auto-generated demos, and feedback collection
Implements comprehensive demonstration capabilities for hackathon judging
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import hashlib
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DemoScenario:
    """Represents a demo scenario for different user personas"""
    scenario_id: str
    title: str
    description: str
    user_persona: str
    difficulty_level: str  # beginner, intermediate, advanced
    estimated_duration: int  # minutes
    steps: List[Dict[str, Any]]
    expected_outcomes: List[str]
    data_sources: List[str]
    confidence_requirements: Dict[str, float]

@dataclass
class UserFeedback:
    """Represents user feedback on AI recommendations"""
    feedback_id: str
    timestamp: datetime
    user_id: str
    recommendation_id: str
    recommendation_type: str
    rating: int  # 1-5 scale
    was_helpful: bool
    was_accurate: bool
    comments: str
    improvement_suggestions: List[str]
    follow_up_actions: List[str]

@dataclass
class DemoMetrics:
    """Tracks demo performance and user engagement"""
    demo_id: str
    start_time: datetime
    end_time: Optional[datetime]
    user_persona: str
    steps_completed: int
    total_steps: int
    time_spent: Optional[float]  # minutes
    confidence_scores: List[float]
    feedback_ratings: List[int]
    data_source_usage: Dict[str, int]
    fallback_usage: int
    success_rate: float

class DemoUserJourney:
    """Manages comprehensive demo capabilities and user journey tracking"""
    
    def __init__(self):
        self.demo_scenarios = self._initialize_demo_scenarios()
        self.current_demo = None
        self.demo_history = []
        self.feedback_database = "demo_feedback.db"
        self.setup_feedback_database()
        
    def _initialize_demo_scenarios(self) -> Dict[str, DemoScenario]:
        """Initialize demo scenarios for different user personas"""
        scenarios = {}
        
        # Farmer Demo Scenario
        scenarios['farmer_basic'] = DemoScenario(
            scenario_id="farmer_basic",
            title="🌾 Farmer Credit Assessment Journey",
            description="Complete journey of a farmer seeking credit assessment with weather and market insights",
            user_persona="Farmer",
            difficulty_level="beginner",
            estimated_duration=8,
            steps=[
                {
                    "step": 1,
                    "title": "Profile Creation",
                    "description": "Create farmer profile with basic information",
                    "input_required": ["name", "location", "land_size", "crop_type"],
                    "expected_output": "Farmer profile created with unique ID"
                },
                {
                    "step": 2,
                    "title": "Weather Risk Assessment",
                    "description": "Analyze weather conditions and risks for the region",
                    "input_required": ["location", "crop_type"],
                    "expected_output": "Weather risk assessment with confidence score"
                },
                {
                    "step": 3,
                    "title": "Market Price Analysis",
                    "description": "Get current market prices and trends for crops",
                    "input_required": ["crop_type", "location"],
                    "expected_output": "Market analysis with price predictions"
                },
                {
                    "step": 4,
                    "title": "Credit Risk Scoring",
                    "description": "AI-powered credit risk assessment",
                    "input_required": ["farmer_profile", "weather_data", "market_data"],
                    "expected_output": "Credit score with detailed explanation"
                },
                {
                    "step": 5,
                    "title": "Government Scheme Matching",
                    "description": "Find relevant government schemes and subsidies",
                    "input_required": ["farmer_profile", "credit_score"],
                    "expected_output": "List of eligible schemes with application details"
                },
                {
                    "step": 6,
                    "title": "Action Plan Generation",
                    "description": "Generate personalized action plan",
                    "input_required": ["all_previous_outputs"],
                    "expected_output": "Comprehensive action plan with timeline"
                }
            ],
            expected_outcomes=[
                "Complete farmer profile with risk assessment",
                "Weather and market insights",
                "AI-powered credit score",
                "Government scheme recommendations",
                "Personalized action plan"
            ],
            data_sources=[
                "OpenWeatherMap API",
                "Agmarknet Market Data",
                "Government Schemes Database",
                "Soil Health Data",
                "Historical Credit Data"
            ],
            confidence_requirements={
                "weather_assessment": 0.8,
                "market_analysis": 0.7,
                "credit_scoring": 0.9,
                "scheme_matching": 0.95
            }
        )
        
        # Financier Demo Scenario
        scenarios['financier_advanced'] = DemoScenario(
            scenario_id="financier_advanced",
            title="🏦 Portfolio Risk Management Journey",
            description="Advanced portfolio analytics and risk management for financial institutions",
            user_persona="Financier",
            difficulty_level="advanced",
            estimated_duration=15,
            steps=[
                {
                    "step": 1,
                    "title": "Portfolio Overview",
                    "description": "Comprehensive portfolio dashboard with key metrics",
                    "input_required": ["portfolio_data"],
                    "expected_output": "Portfolio summary with risk indicators"
                },
                {
                    "step": 2,
                    "title": "Risk Segmentation",
                    "description": "Segment farmers by risk categories and analyze patterns",
                    "input_required": ["farmer_data", "risk_scores"],
                    "expected_output": "Risk segmentation analysis with visualizations"
                },
                {
                    "step": 3,
                    "title": "Weather Risk Correlation",
                    "description": "Analyze correlation between weather patterns and default rates",
                    "input_required": ["weather_data", "default_data"],
                    "expected_output": "Weather-risk correlation analysis"
                },
                {
                    "step": 4,
                    "title": "Market Impact Assessment",
                    "description": "Assess impact of market volatility on portfolio performance",
                    "input_required": ["market_data", "portfolio_performance"],
                    "expected_output": "Market impact assessment with stress testing"
                },
                {
                    "step": 5,
                    "title": "Predictive Analytics",
                    "description": "Generate default risk predictions for portfolio",
                    "input_required": ["historical_data", "current_indicators"],
                    "expected_output": "Risk prediction model with confidence intervals"
                },
                {
                    "step": 6,
                    "title": "Risk Mitigation Strategies",
                    "description": "Develop portfolio-level risk mitigation strategies",
                    "input_required": ["risk_analysis", "market_conditions"],
                    "expected_output": "Comprehensive risk mitigation plan"
                }
            ],
            expected_outcomes=[
                "Portfolio risk assessment",
                "Weather-risk correlation analysis",
                "Market impact assessment",
                "Predictive risk modeling",
                "Risk mitigation strategies"
            ],
            data_sources=[
                "Internal Portfolio Data",
                "Weather APIs",
                "Market Data APIs",
                "Credit Bureau Data",
                "Economic Indicators"
            ],
            confidence_requirements={
                "portfolio_analysis": 0.9,
                "risk_modeling": 0.85,
                "market_analysis": 0.8,
                "predictions": 0.75
            }
        )
        
        # Agricultural Officer Demo Scenario
        scenarios['officer_policy'] = DemoScenario(
            scenario_id="officer_policy",
            title="🏛️ Policy Impact Assessment Journey",
            description="Assess impact of agricultural policies and schemes on farmer outcomes",
            user_persona="Agricultural Officer",
            difficulty_level="intermediate",
            estimated_duration=12,
            steps=[
                {
                    "step": 1,
                    "title": "Policy Overview",
                    "description": "Review current agricultural policies and schemes",
                    "input_required": ["policy_database"],
                    "expected_output": "Policy summary with key objectives"
                },
                {
                    "step": 2,
                    "title": "Farmer Eligibility Analysis",
                    "description": "Analyze farmer eligibility for different schemes",
                    "input_required": ["farmer_data", "policy_criteria"],
                    "expected_output": "Eligibility analysis with coverage statistics"
                },
                {
                    "step": 3,
                    "title": "Scheme Performance Assessment",
                    "description": "Evaluate performance of existing schemes",
                    "input_required": ["scheme_data", "outcome_metrics"],
                    "expected_output": "Performance assessment with success metrics"
                },
                {
                    "step": 4,
                    "title": "Impact Measurement",
                    "description": "Measure impact of schemes on farmer outcomes",
                    "input_required": ["before_after_data", "control_groups"],
                    "expected_output": "Impact measurement with statistical significance"
                },
                {
                    "step": 5,
                    "title": "Policy Recommendations",
                    "description": "Generate policy improvement recommendations",
                    "input_required": ["impact_analysis", "stakeholder_feedback"],
                    "expected_output": "Policy recommendations with implementation plan"
                }
            ],
            expected_outcomes=[
                "Policy performance analysis",
                "Farmer eligibility assessment",
                "Impact measurement",
                "Policy recommendations",
                "Implementation roadmap"
            ],
            data_sources=[
                "Government Policy Database",
                "Farmer Registration Data",
                "Scheme Implementation Data",
                "Outcome Surveys",
                "Economic Indicators"
            ],
            confidence_requirements={
                "eligibility_analysis": 0.9,
                "performance_assessment": 0.8,
                "impact_measurement": 0.75,
                "recommendations": 0.85
            }
        )
        
        return scenarios
    
    def setup_feedback_database(self):
        """Setup database for storing demo feedback and metrics"""
        conn = sqlite3.connect(self.feedback_database)
        cursor = conn.cursor()
        
        # Demo feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS demo_feedback (
            id INTEGER PRIMARY KEY,
            feedback_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            user_id TEXT NOT NULL,
            recommendation_id TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            rating INTEGER NOT NULL,
            was_helpful BOOLEAN NOT NULL,
            was_accurate BOOLEAN NOT NULL,
            comments TEXT,
            improvement_suggestions TEXT,
            follow_up_actions TEXT
        )
        """)
        
        # Demo metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS demo_metrics (
            id INTEGER PRIMARY KEY,
            demo_id TEXT UNIQUE NOT NULL,
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            user_persona TEXT NOT NULL,
            steps_completed INTEGER NOT NULL,
            total_steps INTEGER NOT NULL,
            time_spent REAL,
            confidence_scores TEXT,
            feedback_ratings TEXT,
            data_source_usage TEXT,
            fallback_usage INTEGER,
            success_rate REAL
        )
        """)
        
        # User journey logs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_journey_logs (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_persona TEXT NOT NULL,
            step_number INTEGER NOT NULL,
            step_title TEXT NOT NULL,
            input_data TEXT,
            output_data TEXT,
            confidence_score REAL,
            data_sources TEXT,
            fallback_used BOOLEAN,
            timestamp DATETIME NOT NULL,
            execution_time REAL
        )
        """)
        
        conn.commit()
        conn.close()
    
    def start_demo(self, scenario_id: str, user_persona: str) -> str:
        """Start a new demo session"""
        if scenario_id not in self.demo_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        demo_id = hashlib.md5(f"{datetime.now().isoformat()}{scenario_id}".encode()).hexdigest()[:8]
        
        demo_metrics = DemoMetrics(
            demo_id=demo_id,
            start_time=datetime.now(),
            user_persona=user_persona,
            steps_completed=0,
            total_steps=len(self.demo_scenarios[scenario_id].steps),
            confidence_scores=[],
            feedback_ratings=[],
            data_source_usage={},
            fallback_usage=0,
            success_rate=0.0
        )
        
        self.current_demo = {
            'scenario_id': scenario_id,
            'demo_id': demo_id,
            'current_step': 0,
            'metrics': demo_metrics,
            'start_time': datetime.now()
        }
        
        logger.info(f"Started demo {demo_id} for scenario {scenario_id}")
        return demo_id
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get current demo step information"""
        if not self.current_demo:
            return None
        
        scenario = self.demo_scenarios[self.current_demo['scenario_id']]
        current_step_num = self.current_demo['current_step']
        
        if current_step_num < len(scenario.steps):
            return scenario.steps[current_step_num]
        
        return None
    
    def complete_step(self, step_output: Dict[str, Any], confidence_score: float, 
                     data_sources: List[str], fallback_used: bool = False) -> bool:
        """Complete current demo step and move to next"""
        if not self.current_demo:
            return False
        
        # Update metrics
        self.current_demo['metrics'].steps_completed += 1
        self.current_demo['metrics'].confidence_scores.append(confidence_score)
        
        if fallback_used:
            self.current_demo['metrics'].fallback_usage += 1
        
        # Update data source usage
        for source in data_sources:
            if source in self.current_demo['metrics'].data_source_usage:
                self.current_demo['metrics'].data_source_usage[source] += 1
            else:
                self.current_demo['metrics'].data_source_usage[source] = 1
        
        # Log step completion
        self._log_step_completion(step_output, confidence_score, data_sources, fallback_used)
        
        # Move to next step
        self.current_demo['current_step'] += 1
        
        # Check if demo is complete
        if self.current_demo['current_step'] >= len(self.demo_scenarios[self.current_demo['scenario_id']].steps):
            self._complete_demo()
            return True
        
        return False
    
    def _log_step_completion(self, step_output: Dict[str, Any], confidence_score: float,
                           data_sources: List[str], fallback_used: bool):
        """Log step completion details"""
        if not self.current_demo:
            return
        
        conn = sqlite3.connect(self.feedback_database)
        cursor = conn.cursor()
        
        current_step = self.get_current_step()
        if current_step:
            cursor.execute("""
            INSERT INTO user_journey_logs 
            (session_id, user_persona, step_number, step_title, input_data, output_data, 
             confidence_score, data_sources, fallback_used, timestamp, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.current_demo['demo_id'],
                self.current_demo['metrics'].user_persona,
                current_step['step'],
                current_step['title'],
                json.dumps({}),  # Input data would be captured earlier
                json.dumps(step_output),
                confidence_score,
                json.dumps(data_sources),
                fallback_used,
                datetime.now(),
                0.0  # Execution time would be calculated
            ))
        
        conn.commit()
        conn.close()
    
    def _complete_demo(self):
        """Complete the demo and calculate final metrics"""
        if not self.current_demo:
            return
        
        # Calculate success rate
        total_steps = self.current_demo['metrics'].total_steps
        completed_steps = self.current_demo['metrics'].steps_completed
        success_rate = completed_steps / total_steps if total_steps > 0 else 0.0
        
        # Calculate time spent
        end_time = datetime.now()
        time_spent = (end_time - self.current_demo['start_time']).total_seconds() / 60.0
        
        # Update metrics
        self.current_demo['metrics'].end_time = end_time
        self.current_demo['metrics'].time_spent = time_spent
        self.current_demo['metrics'].success_rate = success_rate
        
        # Store metrics in database
        self._store_demo_metrics(self.current_demo['metrics'])
        
        # Add to history
        self.demo_history.append(self.current_demo.copy())
        
        logger.info(f"Completed demo {self.current_demo['demo_id']} with {success_rate:.1%} success rate")
        
        # Clear current demo
        self.current_demo = None
    
    def _store_demo_metrics(self, metrics: DemoMetrics):
        """Store demo metrics in database"""
        conn = sqlite3.connect(self.feedback_database)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO demo_metrics 
        (demo_id, start_time, end_time, user_persona, steps_completed, total_steps,
         time_spent, confidence_scores, feedback_ratings, data_source_usage,
         fallback_usage, success_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.demo_id,
            metrics.start_time,
            metrics.end_time,
            metrics.user_persona,
            metrics.steps_completed,
            metrics.total_steps,
            metrics.time_spent,
            json.dumps(metrics.confidence_scores),
            json.dumps(metrics.feedback_ratings),
            json.dumps(metrics.data_source_usage),
            metrics.fallback_usage,
            metrics.success_rate
        ))
        
        conn.commit()
        conn.close()
    
    def collect_feedback(self, recommendation_id: str, recommendation_type: str,
                        rating: int, was_helpful: bool, was_accurate: bool,
                        comments: str = "", improvement_suggestions: List[str] = None,
                        follow_up_actions: List[str] = None) -> str:
        """Collect user feedback on AI recommendations"""
        feedback_id = hashlib.md5(f"{datetime.now().isoformat()}{recommendation_id}".encode()).hexdigest()[:8]
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=datetime.now(),
            user_id="demo_user",  # Would be actual user ID in production
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            rating=rating,
            was_helpful=was_helpful,
            was_accurate=was_accurate,
            comments=comments,
            improvement_suggestions=improvement_suggestions or [],
            follow_up_actions=follow_up_actions or []
        )
        
        # Store feedback in database
        conn = sqlite3.connect(self.feedback_database)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO demo_feedback 
        (feedback_id, timestamp, user_id, recommendation_id, recommendation_type,
         rating, was_helpful, was_accurate, comments, improvement_suggestions, follow_up_actions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_id,
            feedback.timestamp,
            feedback.user_id,
            feedback.recommendation_id,
            feedback.recommendation_type,
            feedback.rating,
            feedback.was_helpful,
            feedback.was_accurate,
            feedback.comments,
            json.dumps(feedback.improvement_suggestions),
            json.dumps(feedback.follow_up_actions)
        ))
        
        conn.commit()
        conn.close()
        
        # Update current demo metrics if available
        if self.current_demo:
            self.current_demo['metrics'].feedback_ratings.append(rating)
        
        logger.info(f"Collected feedback {feedback_id} for recommendation {recommendation_id}")
        return feedback_id
    
    def get_demo_progress(self) -> Dict[str, Any]:
        """Get current demo progress information"""
        if not self.current_demo:
            return {"status": "no_demo_active"}
        
        scenario = self.demo_scenarios[self.current_demo['scenario_id']]
        current_step = self.current_demo['current_step']
        total_steps = len(scenario.steps)
        
        progress = {
            "status": "active",
            "scenario_title": scenario.title,
            "user_persona": scenario.user_persona,
            "current_step": current_step + 1,
            "total_steps": total_steps,
            "progress_percentage": ((current_step + 1) / total_steps) * 100,
            "steps_completed": self.current_demo['metrics'].steps_completed,
            "estimated_remaining": (total_steps - current_step - 1) * 2,  # Rough estimate
            "current_step_info": self.get_current_step()
        }
        
        return progress
    
    def generate_demo_report(self, demo_id: str) -> Dict[str, Any]:
        """Generate comprehensive demo report"""
        # Find demo in history
        demo_data = None
        for demo in self.demo_history:
            if demo['demo_id'] == demo_id:
                demo_data = demo
                break
        
        if not demo_data:
            return {"error": "Demo not found"}
        
        metrics = demo_data['metrics']
        scenario = self.demo_scenarios[demo_data['scenario_id']]
        
        # Calculate additional metrics
        avg_confidence = np.mean(metrics.confidence_scores) if metrics.confidence_scores else 0
        avg_feedback = np.mean(metrics.feedback_ratings) if metrics.feedback_ratings else 0
        
        report = {
            "demo_id": demo_id,
            "scenario": {
                "id": demo_data['scenario_id'],
                "title": scenario.title,
                "description": scenario.description,
                "user_persona": scenario.user_persona,
                "difficulty_level": scenario.difficulty_level
            },
            "performance_metrics": {
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "duration_minutes": metrics.time_spent,
                "steps_completed": metrics.steps_completed,
                "total_steps": metrics.total_steps,
                "success_rate": metrics.success_rate,
                "average_confidence": avg_confidence,
                "average_feedback_rating": avg_feedback
            },
            "data_usage": {
                "data_sources": metrics.data_source_usage,
                "fallback_usage": metrics.fallback_usage,
                "data_quality_score": 1.0 - (metrics.fallback_usage / metrics.total_steps) if metrics.total_steps > 0 else 0
            },
            "step_details": self._get_step_details(demo_id),
            "recommendations": self._generate_improvement_recommendations(metrics, scenario),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _get_step_details(self, demo_id: str) -> List[Dict[str, Any]]:
        """Get detailed information about each step in the demo"""
        conn = sqlite3.connect(self.feedback_database)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT step_number, step_title, input_data, output_data, confidence_score, 
               data_sources, fallback_used, timestamp, execution_time
        FROM user_journey_logs 
        WHERE session_id = ? 
        ORDER BY step_number
        """, (demo_id,))
        
        steps = []
        for row in cursor.fetchall():
            steps.append({
                "step_number": row[0],
                "title": row[1],
                "input_data": json.loads(row[2]) if row[2] else {},
                "output_data": json.loads(row[3]) if row[3] else {},
                "confidence_score": row[4],
                "data_sources": json.loads(row[5]) if row[5] else [],
                "fallback_used": bool(row[6]),
                "timestamp": row[7],
                "execution_time": row[8]
            })
        
        conn.close()
        return steps
    
    def _generate_improvement_recommendations(self, metrics: DemoMetrics, 
                                           scenario: DemoScenario) -> List[str]:
        """Generate improvement recommendations based on demo performance"""
        recommendations = []
        
        # Success rate recommendations
        if metrics.success_rate < 0.8:
            recommendations.append("Consider simplifying complex steps or providing more guidance")
        
        # Confidence score recommendations
        if metrics.confidence_scores:
            avg_confidence = np.mean(metrics.confidence_scores)
            if avg_confidence < 0.7:
                recommendations.append("Improve data quality and reduce reliance on fallback data")
        
        # Time efficiency recommendations
        if metrics.time_spent and metrics.time_spent > scenario.estimated_duration * 1.5:
            recommendations.append("Optimize step flow to reduce completion time")
        
        # Data source recommendations
        if metrics.fallback_usage > metrics.total_steps * 0.3:
            recommendations.append("Enhance primary data source reliability and reduce fallback usage")
        
        # General recommendations
        recommendations.extend([
            "Collect more user feedback to identify pain points",
            "Implement A/B testing for different step flows",
            "Add progress indicators and better navigation",
            "Provide contextual help and tooltips"
        ])
        
        return recommendations
    
    def export_demo_data(self, format: str = 'json') -> str:
        """Export demo data in various formats"""
        if format == 'json':
            return json.dumps({
                "demo_history": self.demo_history,
                "scenarios": {k: asdict(v) for k, v in self.demo_scenarios.items()},
                "export_timestamp": datetime.now().isoformat()
            }, default=str, indent=2)
        
        elif format == 'csv':
            # Export demo metrics as CSV
            data = []
            for demo in self.demo_history:
                metrics = demo['metrics']
                data.append({
                    'demo_id': demo['demo_id'],
                    'scenario_id': demo['scenario_id'],
                    'user_persona': metrics.user_persona,
                    'start_time': metrics.start_time,
                    'end_time': metrics.end_time,
                    'steps_completed': metrics.steps_completed,
                    'total_steps': metrics.total_steps,
                    'success_rate': metrics.success_rate,
                    'time_spent_minutes': metrics.time_spent,
                    'fallback_usage': metrics.fallback_usage
                })
            
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get overall demo statistics"""
        if not self.demo_history:
            return {"total_demos": 0}
        
        total_demos = len(self.demo_history)
        successful_demos = sum(1 for demo in self.demo_history if demo['metrics'].success_rate >= 0.8)
        
        # Persona distribution
        persona_counts = {}
        for demo in self.demo_history:
            persona = demo['metrics'].user_persona
            persona_counts[persona] = persona_counts.get(persona, 0) + 1
        
        # Average metrics
        avg_success_rate = np.mean([demo['metrics'].success_rate for demo in self.demo_history])
        avg_time_spent = np.mean([demo['metrics'].time_spent or 0 for demo in self.demo_history])
        avg_confidence = np.mean([
            np.mean(demo['metrics'].confidence_scores) 
            for demo in self.demo_history 
            if demo['metrics'].confidence_scores
        ])
        
        return {
            "total_demos": total_demos,
            "successful_demos": successful_demos,
            "success_rate": successful_demos / total_demos if total_demos > 0 else 0,
            "persona_distribution": persona_counts,
            "average_success_rate": avg_success_rate,
            "average_time_spent_minutes": avg_time_spent,
            "average_confidence": avg_confidence,
            "last_demo": max([demo['metrics'].start_time for demo in self.demo_history]).isoformat() if self.demo_history else None
        }

# Global instance
demo_manager = DemoUserJourney()
