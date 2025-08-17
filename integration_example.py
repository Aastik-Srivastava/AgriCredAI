"""
Integration Example for Refactored AgriCredAI
Demonstrates how to use all the new modules together
"""

import streamlit as st
from public_data_integration import data_manager
from multilingual_multimodal import multimodal_manager
from explainable_ai_core import explainable_ai
from offline_edge_support import offline_support
from demo_user_journey import demo_manager
import pandas as pd
from datetime import datetime

def main():
    st.title("🌾 AgriCredAI - Refactored System Integration Demo")
    st.markdown("### Demonstrating Public Data, Multi-lingual Support, Explainable AI, and Offline Capabilities")
    
    # Sidebar for demo selection
    st.sidebar.header("🎯 Demo Selection")
    demo_type = st.sidebar.selectbox(
        "Choose Demo Type",
        ["Data Integration", "Multi-lingual", "Explainable AI", "Offline Support", "User Journey"]
    )
    
    if demo_type == "Data Integration":
        show_data_integration_demo()
    elif demo_type == "Multi-lingual":
        show_multilingual_demo()
    elif demo_type == "Explainable AI":
        show_explainable_ai_demo()
    elif demo_type == "Offline Support":
        show_offline_support_demo()
    elif demo_type == "User Journey":
        show_user_journey_demo()

def show_data_integration_demo():
    st.header("🌐 Public Data Integration Demo")
    st.markdown("Demonstrating real data sources and fallback systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location Selection")
        location = st.selectbox(
            "Select Region",
            ["Punjab", "Maharashtra", "Uttar Pradesh", "Karnataka", "Andhra Pradesh"]
        )
        
        if st.button("🌦️ Get Weather Data"):
            with st.spinner("Fetching weather data..."):
                # Get weather data with fallback
                weather_data = data_manager.get_weather_data(28.6139, 77.2090, location)
                
                st.success("✅ Weather data retrieved!")
                st.json({
                    "source": weather_data.source.name,
                    "confidence": f"{weather_data.confidence_score:.1%}",
                    "fallback_used": weather_data.fallback_used,
                    "data_freshness": weather_data.data_provenance["data_freshness"]
                })
                
                # Display weather info
                if weather_data.data:
                    temp = weather_data.data.get('main', {}).get('temp', 'N/A')
                    humidity = weather_data.data.get('main', {}).get('humidity', 'N/A')
                    st.metric("Temperature", f"{temp}°C")
                    st.metric("Humidity", f"{humidity}%")
    
    with col2:
        st.subheader("🌾 Market Data")
        commodity = st.selectbox(
            "Select Commodity",
            ["wheat", "rice", "cotton", "sugarcane", "soybean"]
        )
        
        if st.button("💹 Get Market Data"):
            with st.spinner("Fetching market data..."):
                market_data = data_manager.get_market_data(commodity, location)
                
                if market_data.data and market_data.data.get('prices'):
                    price = market_data.data['prices'][0].get('modal_price', 'N/A')
                    st.success(f"✅ {commodity.title()} price: ₹{price}/quintal")
                    st.info(f"Source: {market_data.source.name}")
                    st.info(f"Confidence: {market_data.confidence_score:.1%}")
                else:
                    st.warning("No market data available")
    
    # Data source summary
    st.subheader("📊 Data Source Summary")
    summary = data_manager.export_data_summary()
    st.json(summary)

def show_multilingual_demo():
    st.header("🗣️ Multi-lingual & Multi-modal Demo")
    st.markdown("Demonstrating language support and voice capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Language Support")
        languages = multimodal_manager.get_supported_languages()
        
        selected_lang = st.selectbox(
            "Select Language",
            languages,
            format_func=lambda x: f"{x['native_name']} ({x['name']})"
        )
        
        if selected_lang:
            st.info(f"Selected: {selected_lang['native_name']}")
            
            # Language detection demo
            st.subheader("🔍 Language Detection")
            text_input = st.text_input("Enter text in any supported language:")
            
            if text_input:
                detected_lang, confidence = multimodal_manager.detect_language(text_input)
                st.success(f"Detected: {multimodal_manager.get_language_display_name(detected_lang)}")
                st.info(f"Confidence: {confidence:.1%}")
    
    with col2:
        st.subheader("🎤 Voice Input Demo")
        st.info("Voice input requires microphone access")
        
        # Simulate voice input
        if st.button("🎙️ Simulate Voice Input"):
            # Create a mock voice input
            mock_audio = b"mock_audio_data"
            voice_input = multimodal_manager.process_voice_input(mock_audio)
            
            st.success("Voice input processed!")
            st.json({
                "language_detected": voice_input.language_detected,
                "transcription": voice_input.transcription or "No transcription",
                "confidence": f"{voice_input.confidence:.1%}"
            })
    
    # Multi-modal query demo
    st.subheader("🔍 Multi-modal Query Demo")
    query_text = st.text_input("Enter your query:")
    
    if query_text and st.button("🔍 Process Query"):
        # Create multi-modal query
        query = multimodal_manager.create_multimodal_query(text=query_text)
        intent, entities = multimodal_manager.extract_intent_and_entities(query)
        
        st.success("Query processed!")
        st.info(f"Intent: {intent}")
        st.info(f"Entities: {len(entities)} found")
        
        if entities:
            st.json(entities)

def show_explainable_ai_demo():
    st.header("🧠 Explainable AI Demo")
    st.markdown("Demonstrating confidence scoring and explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confidence Scoring")
        
        # Simulate different confidence factors
        data_quality = st.slider("Data Quality", 0.0, 1.0, 0.8)
        model_performance = st.slider("Model Performance", 0.0, 1.0, 0.9)
        feature_completeness = st.slider("Feature Completeness", 0.0, 1.0, 0.85)
        temporal_relevance = st.slider("Temporal Relevance", 0.0, 1.0, 0.9)
        spatial_coverage = st.slider("Spatial Coverage", 0.0, 1.0, 0.8)
        
        if st.button("🎯 Calculate Confidence"):
            confidence_breakdown = explainable_ai.calculate_confidence_score(
                data_quality, model_performance, feature_completeness,
                temporal_relevance, spatial_coverage
            )
            
            st.success(f"Overall Confidence: {confidence_breakdown.overall_confidence:.1%}")
            
            # Display breakdown
            st.subheader("Confidence Breakdown")
            for factor, score in confidence_breakdown.factors.items():
                st.metric(factor.replace('_', ' ').title(), f"{score:.1%}")
    
    with col2:
        st.subheader("📝 AI Explanation Demo")
        
        # Simulate farmer data
        farmer_data = {
            'farmer_age': 45,
            'land_size': 3.5,
            'education_level': 3,
            'irrigation_access': 1,
            'payment_history_score': 0.85,
            'debt_to_income_ratio': 0.3,
            'soil_health_index': 0.8
        }
        
        if st.button("👨‍🌾 Generate Credit Explanation"):
            # Generate explanation (simplified for demo)
            explanation = explainable_ai.generate_credit_explanation(
                farmer_data, 0.25, 0.85, None, list(farmer_data.keys())
            )
            
            st.success("Explanation generated!")
            st.info(f"Decision: {explanation.decision_summary}")
            
            # Show key factors
            st.subheader("Key Factors")
            for factor in explanation.key_factors[:3]:
                st.write(f"• {factor['feature']}: {factor['value']} ({factor['impact']})")

def show_offline_support_demo():
    st.header("📱 Offline & Edge Support Demo")
    st.markdown("Demonstrating offline capabilities and fallback systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Data Caching")
        
        # Cache some sample data
        if st.button("💾 Cache Sample Data"):
            sample_data = {"temperature": 28, "humidity": 65, "source": "demo"}
            cache_hash = offline_support.cache_data(
                "weather_demo", sample_data, "demo_source", 0.9, 1
            )
            st.success(f"Data cached with hash: {cache_hash[:8]}")
        
        # Retrieve cached data
        if st.button("📥 Retrieve Cached Data"):
            cached_data = offline_support.get_cached_data("weather_demo", max_age_hours=2)
            if cached_data:
                st.success("Cached data retrieved!")
                st.json(cached_data)
            else:
                st.warning("No cached data available")
    
    with col2:
        st.subheader("📡 Offline Queries")
        
        # Create offline query
        if st.button("📝 Create Offline Query"):
            query_id = offline_support.create_offline_query(
                "weather_inquiry",
                {"location": "Punjab", "crop_type": "wheat"},
                farmer_id="FARM001",
                priority="normal"
            )
            st.success(f"Offline query created: {query_id}")
        
        # Show pending queries
        if st.button("📋 Show Pending Queries"):
            pending = offline_support.get_pending_offline_queries()
            if pending:
                st.success(f"Found {len(pending)} pending queries")
                for query in pending[:3]:
                    st.info(f"Query {query.query_id}: {query.query_type}")
            else:
                st.info("No pending queries")
    
    # Offline data demo
    st.subheader("🌍 Offline Data Access")
    region = st.selectbox("Select Region for Offline Data", ["Punjab", "Maharashtra", "UP", "Karnataka"])
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("🌦️ Offline Weather"):
            weather = offline_support.get_offline_weather_data(region)
            if weather:
                st.success("Offline weather data available!")
                st.metric("Temperature", f"{weather['main']['temp']}°C")
            else:
                st.warning("No offline weather data")
    
    with col4:
        if st.button("💹 Offline Market"):
            market = offline_support.get_offline_market_data("wheat", region)
            if market:
                st.success("Offline market data available!")
                st.metric("Price", f"₹{market['price']}/qt")
            else:
                st.warning("No offline market data")
    
    with col5:
        if st.button("🌱 Offline Soil"):
            soil = offline_support.get_offline_soil_data(region)
            if soil:
                st.success("Offline soil data available!")
                st.metric("pH", f"{soil['ph']}")
            else:
                st.warning("No offline soil data")

def show_user_journey_demo():
    st.header("👥 User Journey Demo")
    st.markdown("Demonstrating interactive demo scenarios and tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Demo Scenarios")
        
        # Select scenario
        scenario = st.selectbox(
            "Choose Demo Scenario",
            list(demo_manager.demo_scenarios.keys()),
            format_func=lambda x: demo_manager.demo_scenarios[x].title
        )
        
        if scenario:
            scenario_info = demo_manager.demo_scenarios[scenario]
            st.info(f"**{scenario_info.title}**")
            st.write(scenario_info.description)
            st.write(f"**Difficulty:** {scenario_info.difficulty_level}")
            st.write(f"**Duration:** {scenario_info.estimated_duration} minutes")
            st.write(f"**Steps:** {len(scenario_info.steps)}")
    
    with col2:
        st.subheader("🚀 Start Demo")
        
        if st.button("▶️ Start Demo Session"):
            demo_id = demo_manager.start_demo(scenario, "demo_user")
            st.success(f"Demo started! ID: {demo_id}")
            
            # Store demo ID in session state
            st.session_state.demo_id = demo_id
    
    # Demo progress
    if hasattr(st.session_state, 'demo_id'):
        st.subheader("📊 Demo Progress")
        progress = demo_manager.get_demo_progress()
        
        if progress['status'] == 'active':
            st.progress(progress['progress_percentage'] / 100)
            st.info(f"Step {progress['current_step']} of {progress['total_steps']}")
            
            current_step = progress['current_step_info']
            if current_step:
                st.write(f"**Current Step:** {current_step['title']}")
                st.write(f"**Description:** {current_step['description']}")
                
                # Simulate step completion
                if st.button("✅ Complete Step"):
                    # Simulate step output
                    step_output = {
                        "status": "completed",
                        "confidence": 0.85,
                        "data_sources": ["api", "cache"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    demo_manager.complete_step(
                        step_output, 0.85, ["api", "cache"], False
                    )
                    st.success("Step completed!")
                    st.rerun()
        else:
            st.info("No active demo")
    
    # Demo statistics
    st.subheader("📈 Demo Statistics")
    stats = demo_manager.get_demo_statistics()
    st.json(stats)

if __name__ == "__main__":
    main()
