# AGENTIC AI DEMO INTERFACE
# Streamlit interface to showcase the agentic AI system in action

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import json

# Import the agentic components
from agentic_core import AgenticOrchestrator, BaseAgent
from dynamic_financing_agent import DynamicFinancingAgent
from carbon_credit_agent import CarbonCreditAgent  
from market_advisory_agent import MarketAdvisoryAgent

def initialize_agentic_system():
    """Initialize the complete agentic AI system"""
    if 'orchestrator' not in st.session_state:
        orchestrator = AgenticOrchestrator()
        
        # Register all agents
        orchestrator.register_agent(DynamicFinancingAgent())
        orchestrator.register_agent(CarbonCreditAgent())
        orchestrator.register_agent(MarketAdvisoryAgent())
        
        st.session_state.orchestrator = orchestrator
        st.session_state.agent_results = {}

async def run_agentic_workflow(farmer_context):
    """Run the complete agentic workflow"""
    orchestrator = st.session_state.orchestrator
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run all agents in parallel
    status_text.text("🔍 Agents analyzing farmer profile...")
    progress_bar.progress(25)
    
    # Execute multi-agent workflow
    results = await orchestrator.run_multi_agent_workflow(farmer_context)
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    return results

def display_agent_reasoning(agent_name, action):
    """Display the reasoning trace of an agent"""
    st.subheader(f"🧠 {agent_name} Reasoning Process")
    
    if action and hasattr(action, 'reasoning_trace'):
        for step in action.reasoning_trace:
            if step.strip():  # Skip empty lines
                st.write(step)
    
    # Show confidence and execution time
    if action:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence Score", f"{action.confidence_score:.1%}")
        with col2:
            st.metric("Execution Time", f"{action.execution_time_ms}ms")

def display_financing_results(action):
    """Display financing agent results"""
    if not action or not hasattr(action, 'outputs'):
        st.error("No financing results available")
        return
        
    financing_package = action.outputs.get('financing_package', {})
    
    st.success(f"💰 **Financing Recommendation:** {action.outputs.get('recommendation_summary', 'No summary')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Loan Amount", f"₹{financing_package.get('loan_amount', 0):,.0f}")
    with col2:
        st.metric("Interest Rate", f"{financing_package.get('interest_rate', 0):.2f}%")
    with col3:
        st.metric("Tenure", f"{financing_package.get('tenure_months', 0)} months")
    with col4:
        st.metric("Approval Probability", f"{financing_package.get('approval_probability', 0):.1%}")
    
    # Loan details
    with st.expander("📋 Detailed Loan Package"):
        st.json(financing_package)
    
    # Repayment schedule
    if 'repayment_schedule' in financing_package:
        st.subheader("📅 Adaptive Repayment Schedule")
        schedule_df = pd.DataFrame(financing_package['repayment_schedule'])
        st.dataframe(schedule_df, use_container_width=True)

def display_carbon_credit_results(action):
    """Display carbon credit agent results"""
    if not action or not hasattr(action, 'outputs'):
        st.error("No carbon credit results available")
        return
        
    credit_certificate = action.outputs.get('credit_certificate', {})
    trading_strategy = action.outputs.get('trading_strategy', {})
    
    st.success(f"🌱 **Carbon Credits:** {action.outputs.get('summary', 'No summary')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CO₂ Sequestration", f"{credit_certificate.get('sequestration_amount', 0):.2f} tons")
    with col2:
        st.metric("Credit Value", f"₹{credit_certificate.get('estimated_value_usd', 0):,.0f}")
    with col3:
        st.metric("Quality Tier", credit_certificate.get('quality_tier', 'Unknown'))
    with col4:
        st.metric("Market Ready", "✅" if credit_certificate.get('market_readiness', False) else "⏳")
    
    # Blockchain certificate
    st.subheader("🔐 Blockchain Certificate")
    st.code(f"Credit ID: {credit_certificate.get('credit_id', 'Unknown')}\nBlockchain Hash: {credit_certificate.get('blockchain_hash', 'Unknown')}")
    
    # Buyer matches
    if trading_strategy.get('buyer_matches'):
        st.subheader("🎯 Potential Buyers")
        buyers_df = pd.DataFrame(trading_strategy['buyer_matches'])
        st.dataframe(buyers_df, use_container_width=True)

def display_market_advisory_results(action):
    """Display market advisory agent results"""
    if not action or not hasattr(action, 'outputs'):
        st.error("No market advisory results available")
        return
        
    market_advisory = action.outputs.get('market_advisory', {})
    recommendation = market_advisory.get('primary_recommendation', {})
    
    st.success(f"📊 **Market Advisory:** {action.outputs.get('summary', 'No summary')}")
    
    # Key recommendation
    st.subheader(f"🎯 Recommendation: {recommendation.get('action', 'Unknown')}")
    st.write(f"**Reason:** {recommendation.get('reason', 'No reason provided')}")
    st.write(f"**Urgency:** {recommendation.get('urgency', 'Unknown')}")
    
    # Price forecast
    price_forecast = market_advisory.get('price_forecast', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price Direction", price_forecast.get('direction', 'Unknown'))
    with col2:
        st.metric("Expected Change", f"{price_forecast.get('magnitude', 0):+.1%}")
    with col3:
        st.metric("Target Price", f"₹{price_forecast.get('target_price', 0):.0f}")
    
    # Action plan
    if market_advisory.get('action_plan'):
        st.subheader("📋 Action Plan")
        for step in market_advisory['action_plan']:
            st.write(f"**Step {step.get('step', '?')}:** {step.get('action', 'Unknown')} - *{step.get('timeline', 'Unknown')}*")

def agentic_ai_demo():
    """Main demo interface"""
    st.title("🤖 Agentic AI System - Live Demo")
    st.markdown("**Experience the power of autonomous AI agents working together to provide comprehensive agricultural intelligence.**")
    
    # Initialize system
    initialize_agentic_system()
    
    # Demo farmer profiles
    demo_farmers = {
        "Raj Singh (Punjab)": {
            'farmer_id': 'FARMER_001',
            'name': 'Raj Singh', 
            'land_size': 5.2,
            'crop_type': 'Wheat',
            'region': 'Punjab',
            'latitude': 30.7333,
            'longitude': 76.7794,
            'irrigation_access': 1,
            'insurance_coverage': 1,
            'cooperative_membership': 1,
            'organic_farming': True,
            'technology_adoption': 0.8
        },
        "Priya Patel (Maharashtra)": {
            'farmer_id': 'FARMER_002', 
            'name': 'Priya Patel',
            'land_size': 3.1,
            'crop_type': 'Cotton',
            'region': 'Maharashtra',
            'latitude': 19.7515,
            'longitude': 75.7139,
            'irrigation_access': 0,
            'insurance_coverage': 0,
            'cooperative_membership': 1,
            'organic_farming': False,
            'technology_adoption': 0.4
        },
        "Suresh Kumar (UP)": {
            'farmer_id': 'FARMER_003',
            'name': 'Suresh Kumar',
            'land_size': 2.8,
            'crop_type': 'Rice', 
            'region': 'UP',
            'latitude': 26.8467,
            'longitude': 80.9462,
            'irrigation_access': 1,
            'insurance_coverage': 0,
            'cooperative_membership': 0,
            'organic_farming': False,
            'technology_adoption': 0.6
        }
    }
    
    # Farmer selection
    st.sidebar.header("👨‍🌾 Select Demo Farmer")
    selected_farmer = st.sidebar.selectbox(
        "Choose a farmer profile:",
        list(demo_farmers.keys())
    )
    
    farmer_context = {
        'farmer_profile': demo_farmers[selected_farmer],
        'credit_assessment': {
            'credit_score': 650,
            'payment_history_score': 0.82,
            'debt_to_income_ratio': 0.45
        }
    }
    
    # Display selected farmer info
    st.sidebar.subheader("📋 Farmer Profile")
    farmer = demo_farmers[selected_farmer]
    st.sidebar.write(f"**Land:** {farmer['land_size']} hectares")
    st.sidebar.write(f"**Crop:** {farmer['crop_type']}")
    st.sidebar.write(f"**Region:** {farmer['region']}")
    st.sidebar.write(f"**Irrigation:** {'✅' if farmer['irrigation_access'] else '❌'}")
    st.sidebar.write(f"**Insurance:** {'✅' if farmer['insurance_coverage'] else '❌'}")
    
    # Run analysis button
    if st.button("🚀 **Run Agentic Analysis**", type="primary"):
        st.markdown("---")
        st.subheader("🔄 Multi-Agent Analysis in Progress...")
        
        # Run the agentic workflow
        try:
            results = asyncio.run(run_agentic_workflow(farmer_context))
            st.session_state.agent_results = results
            
            st.success("🎉 **Analysis Complete!** All agents have finished their analysis.")
            
        except Exception as e:
            st.error(f"Error running agents: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.agent_results:
        st.markdown("---")
        st.header("📊 Agentic AI Results")
        
        results = st.session_state.agent_results
        
        # Create tabs for each agent
        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Dynamic Financing", 
            "🌱 Carbon Credits", 
            "📊 Market Advisory",
            "🧠 Agent Reasoning"
        ])
        
        with tab1:
            st.subheader("Dynamic Financing Agent Results")
            financing_action = results.get('DynamicFinancingAgent')
            if financing_action:
                display_financing_results(financing_action)
            else:
                st.error("Financing agent results not available")
        
        with tab2:
            st.subheader("Carbon Credit Agent Results")
            carbon_action = results.get('CarbonCreditAgent')
            if carbon_action:
                display_carbon_credit_results(carbon_action)
            else:
                st.error("Carbon credit agent results not available")
        
        with tab3:
            st.subheader("Market Advisory Agent Results")
            market_action = results.get('MarketAdvisoryAgent')
            if market_action:
                display_market_advisory_results(market_action)
            else:
                st.error("Market advisory agent results not available")
        
        with tab4:
            st.subheader("Agent Reasoning Traces")
            
            # Show reasoning for each agent
            for agent_name, action in results.items():
                if action and hasattr(action, 'reasoning_trace'):
                    with st.expander(f"🧠 {agent_name} Reasoning"):
                        display_agent_reasoning(agent_name, action)
        
        # Overall summary
        st.markdown("---")
        st.subheader("🎯 Integrated Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏦 Financing", "Approved" if results.get('DynamicFinancingAgent') else "Pending")
            
        with col2:  
            carbon_action = results.get('CarbonCreditAgent')
            co2_amount = 0
            if carbon_action and hasattr(carbon_action, 'outputs'):
                co2_amount = carbon_action.outputs.get('credit_certificate', {}).get('sequestration_amount', 0)
            st.metric("🌱 Carbon Credits", f"{co2_amount:.1f} tCO₂")
            
        with col3:
            market_action = results.get('MarketAdvisoryAgent') 
            market_signal = "Analyzing..."
            if market_action and hasattr(market_action, 'outputs'):
                advisory = market_action.outputs.get('market_advisory', {})
                recommendation = advisory.get('primary_recommendation', {})
                market_signal = recommendation.get('action', 'Unknown')
            st.metric("📊 Market Signal", market_signal)

# Streamlit app configuration
if __name__ == "__main__":
    st.set_page_config(
        page_title="AgriCred AI - Agentic Demo",
        page_icon="🤖", 
        layout="wide"
    )
    
    agentic_ai_demo()