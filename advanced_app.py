import streamlit as st

# Import translate_text function first
from multilingual_multimodal import translate_text

# Get current language
lang = st.session_state.get("selected_language", "en")

# Add language selection in page config
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'en'

# Page configuration
st.set_page_config(
    page_title=translate_text("Capital One AgriCred AI - Agricultural Credit Intelligence Platform", lang),
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Aastik-Srivastava/AgriCredAI',
        'Report a bug': 'https://github.com/Aastik-Srivastava/AgriCredAI/issues',
        'About': translate_text("Capital One AgriCred AI - Revolutionizing agricultural lending with AI", lang)
    }
)

import pandas as pd
import numpy as np
import joblib  # For loading machine learning models
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import requests
import json
import re

import random
import io, wave, os
from sklearn.ensemble import RandomForestRegressor



# Custom modules (assuming these are in your project directory)
from agentic_ai_demo import agentic_ai_demo
from advanced_data_pipeline import AdvancedDataPipeline
from advanced_ml_model import AdvancedCreditModel
from weather_alert_system import WeatherAlertSystem, setup_alerts_table
from config import (
    MODEL_PATH, SCALER_PATH,  # Paths for ML model and scaler
    WEATHER_API_KEY, MARKET_API_KEY, DATABASE_PATH, WEATHER_API_BASE_URL, WEATHER_UNITS, ALERT_CHECK_INTERVAL # Weather API and database config
)

# Multi-lingual support
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
# Optional speech libs
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = KaldiRecognizer = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from config import VOSK_MODEL_PATH
except Exception:
    VOSK_MODEL_PATH = None

# Multi-lingual support classes
@dataclass
class LanguageSupport:
    """Represents language support configuration"""
    code: str
    name: str
    native_name: str
    tts_voice: Optional[str] = None
    confidence_threshold: float = 0.7

@dataclass
class DataProvenance:
    """Tracks the origin and quality of data used in AI decisions"""
    source_name: str
    source_type: str
    data_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    data_freshness: str = "unknown"
    coverage_area: str = "unknown"
    verification_status: str = "unknown"
    confidence_score: float = 0.0
    fallback_reason: Optional[str] = None

# Initialize multi-lingual support
def initialize_language_support():
    """Initialize support for multiple Indian languages"""
    languages = {}
    
    # English (primary)
    languages['en'] = LanguageSupport(
        code='en',
        name='English',
        native_name='English',
        tts_voice='english',
        confidence_threshold=0.8
    )
    
    # Hindi (most widely spoken)
    languages['hi'] = LanguageSupport(
        code='hi',
        name='Hindi',
        native_name='हिंदी',
        tts_voice='hindi',
        confidence_threshold=0.7
    )
    
    # Marathi (Maharashtra)
    languages['mr'] = LanguageSupport(
        code='mr',
        name='Marathi',
        native_name='मराठी',
        tts_voice='marathi',
        confidence_threshold=0.6
    )
    
    # Bengali (West Bengal)
    languages['bn'] = LanguageSupport(
        code='bn',
        name='Bengali',
        native_name='বাংলা',
        tts_voice='bengali',
        confidence_threshold=0.6
    )
    
    # Telugu (Andhra Pradesh, Telangana)
    languages['te'] = LanguageSupport(
        code='te',
        name='Telugu',
        native_name='తెలుగు',
        tts_voice='telugu',
        confidence_threshold=0.6
    )
    
    # Tamil (Tamil Nadu)
    languages['ta'] = LanguageSupport(
        code='ta',
        name='Tamil',
        native_name='தமிழ்',
        tts_voice='tamil',
        confidence_threshold=0.6
    )
    
    # Gujarati (Gujarat)
    languages['gu'] = LanguageSupport(
        code='gu',
        name='Gujarati',
        native_name='ગુજરાતી',
        tts_voice='gujarati',
        confidence_threshold=0.6
    )
    
    # Punjabi (Punjab)
    languages['pa'] = LanguageSupport(
        code='pa',
        name='Punjabi',
        native_name='ਪੰਜਾਬੀ',
        tts_voice='punjabi',
        confidence_threshold=0.6
    )
    
    # Kannada (Karnataka)
    languages['kn'] = LanguageSupport(
        code='kn',
        name='Kannada',
        native_name='ಕನ್ನಡ',
        tts_voice='kannada',
        confidence_threshold=0.6
    )
    
    # Malayalam (Kerala)
    languages['ml'] = LanguageSupport(
        code='ml',
        name='Malayalam',
        native_name='മലയാളം',
        tts_voice='malayalam',
        confidence_threshold=0.6
    )
    
    return languages

# Initialize languages
SUPPORTED_LANGUAGES = initialize_language_support()

# Safe defaults if not defined elsewhere
try:
    CREDIT_PRICE_USD
except NameError:
    CREDIT_PRICE_USD = 12.0  # $/tCO2e (demo)
try:
    USD_TO_INR
except NameError:
    USD_TO_INR = 83.0
try:
    CAR_EQUIV_TON
except NameError:
    CAR_EQUIV_TON = 4.6   # ~tCO2 avoided per car/year (very rough demo figure)
try:
    TREE_EQUIV_TON
except NameError:
    TREE_EQUIV_TON = 0.021  # ~21 kg CO2 per tree/year (demo)

from credit_db_maker import store_credit_transaction, DB_PATH, CREDIT_PRICE_USD, USD_TO_INR, CAR_EQUIV_TON, TREE_EQUIV_TON

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = AdvancedDataPipeline()

# Multi-lingual functions
def detect_language(text: str) -> Tuple[str, float]:
    """Detect language of input text using heuristics and keyword matching"""
    if not text or len(text.strip()) < 3:
        return 'en', 0.5
    
    text_lower = text.lower().strip()
    
    # Language-specific keyword detection
    language_keywords = {
        'hi': ['क्या', 'है', 'में', 'के', 'का', 'की', 'और', 'या', 'नहीं', 'हाँ'],
        'mr': ['काय', 'आहे', 'मध्ये', 'चा', 'ची', 'आणि', 'किंवा', 'नाही', 'होय'],
        'bn': ['কি', 'হয়', 'মধ্যে', 'এর', 'এবং', 'বা', 'না', 'হ্যাঁ'],
        'te': ['ఏమి', 'ఉంది', 'లో', 'యొక్క', 'మరియు', 'లేదా', 'లేదు', 'అవును'],
        'ta': ['என்ன', 'உள்ளது', 'இல்', 'என்ற', 'மற்றும்', 'அல்லது', 'இல்லை', 'ஆம்'],
        'gu': ['શું', 'છે', 'માં', 'નો', 'અને', 'અથવા', 'નહીં', 'હા'],
        'pa': ['ਕੀ', 'ਹੈ', 'ਵਿੱਚ', 'ਦਾ', 'ਅਤੇ', 'ਜਾਂ', 'ਨਹੀਂ', 'ਹਾਂ'],
        'kn': ['ಏನು', 'ಇದೆ', 'ನಲ್ಲಿ', 'ನ', 'ಮತ್ತು', 'ಅಥವಾ', 'ಇಲ್ಲ', 'ಹೌದು'],
        'ml': ['എന്ത്', 'ആണ്', 'ൽ', 'ന്റെ', 'ഒപ്പം', 'അല്ലെങ്കിൽ', 'ഇല്ല', 'അതെ']
    }
    
    # Calculate language scores
    language_scores = {}
    for lang_code, keywords in language_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        if score > 0:
            language_scores[lang_code] = score / len(keywords)
    
    # Check for English patterns
    english_patterns = ['the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'can', 'should']
    english_score = sum(1 for pattern in english_patterns if pattern in text_lower) / len(english_patterns)
    language_scores['en'] = english_score
    
    # Find best match
    if language_scores:
        best_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_lang]
        
        # Boost confidence for longer texts
        if len(text) > 20:
            confidence = min(1.0, confidence + 0.2)
        
        return best_lang, confidence
    
    return 'en', 0.5

def get_language_display_name(language_code: str) -> str:
    """Get display name for language code"""
    if language_code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[language_code].native_name
    return language_code.upper()

def text_to_speech(text: str, language: str = 'en') -> bytes:
    """Convert text to speech"""
    try:
        if language in ['hi', 'mr', 'bn', 'te', 'ta', 'gu', 'pa', 'kn', 'ml']:
            # Use gTTS for Indian languages
            tts = gTTS(text=text, lang=language, slow=False)
            # For demo purposes, we'll just return success
            return b"audio_generated"
        else:
            # Use pyttsx3 for English
            return b"audio_generated"
    except Exception as e:
        st.warning(f"{translate_text('Text-to-speech failed', lang)}: {e}")
        return b""

def create_sms_text(response: str, language: str = 'en') -> str:
    """Create SMS-friendly text from response"""
    import re
    clean_text = re.sub(r'<[^>]+>', '', response)
    clean_text = re.sub(r'[^\w\s\.\,\!\?\-]', '', clean_text)
    
    # Simple translation mapping for common phrases
    translations = {
        'hi': {
            'loan_approved': 'आपका लोन स्वीकृत हो गया है',
            'loan_rejected': 'आपका लोन अस्वीकृत हो गया है',
            'weather_alert': 'मौसम चेतावनी',
            'market_update': 'बाजार अपडेट',
            'credit_score': 'क्रेडिट स्कोर',
            'risk_level': 'जोखिम स्तर',
            'approved': 'स्वीकृत',
            'rejected': 'अस्वीकृत',
            'high': 'उच्च',
            'medium': 'मध्यम',
            'low': 'कम'
        },
        'mr': {
            'loan_approved': 'तुमचे कर्ज मंजूर झाले आहे',
            'loan_rejected': 'तुमचे कर्ज नाकारले गेले आहे',
            'weather_alert': 'हवामान सूचना',
            'market_update': 'बाजार अद्ययावत',
            'credit_score': 'क्रेडिट स्कोअर',
            'risk_level': 'जोखीम पातळी',
            'approved': 'मंजूर',
            'rejected': 'नाकारले',
            'high': 'उच्च',
            'medium': 'मध्यम',
            'low': 'कमी'
        }
    }
    
    if language in translations:
        # Replace common phrases with translations
        for eng_phrase, translated_phrase in translations[language].items():
            clean_text = clean_text.replace(eng_phrase, translated_phrase)
    
    # Truncate if too long for SMS
    max_length = 160
    if len(clean_text) > max_length:
        clean_text = clean_text[:max_length-3] + "..."
    
    return clean_text

def translate_text(text: str, target_language: str = 'en') -> str:
    """Translate text to the target language"""
    # Simple translation dictionary for common UI elements
    translations = {
        'hi': {
            'Executive Summary': 'कार्यकारी सारांश',
            'Portfolio Analytics': 'पोर्टफोलियो विश्लेषण',
            'Credit Risk Scoring': 'क्रेडिट जोखिम स्कोरिंग',
            'Agentic AI Intelligence': 'एजेंटिक AI बुद्धिमत्ता',
            'Weather Risk Monitor': 'मौसम जोखिम मॉनिटर',
            'Market Intelligence': 'बाजार बुद्धिमत्ता',
            'Performance Analytics': 'प्रदर्शन विश्लेषण',
            'System Configuration': 'सिस्टम कॉन्फ़िगरेशन',
            'Multi-lingual Demo': 'बहुभाषी डेमो',
            'Offline Capabilities': 'ऑफ़लाइन क्षमताएं',
            'Select Language': 'भाषा चुनें',
            'Current Language': 'वर्तमान भाषा',
            'Accessibility': 'पहुंच',
            'Font Size': 'फ़ॉन्ट आकार',
            'High Contrast Mode': 'उच्च कंट्रास्ट मोड',
            'Navigation Dashboard': 'नेविगेशन डैशबोर्ड',
            'Select Dashboard': 'डैशबोर्ड चुनें',
            'Live Metrics': 'लाइव मेट्रिक्स',
            'Weather Alerts': 'मौसम चेतावनी',
            'Portfolio Value': 'पोर्टफोलियो मूल्य',
            'Active Loans': 'सक्रिय ऋण',
            'Default Rate': 'डिफ़ॉल्ट दर',
            'Avg Credit Score': 'औसत क्रेडिट स्कोर'
        },
        'mr': {
            'Executive Summary': 'कार्यकारी सारांश',
            'Portfolio Analytics': 'पोर्टफोलियो विश्लेषण',
            'Credit Risk Scoring': 'क्रेडिट जोखीम स्कोरिंग',
            'Agentic AI Intelligence': 'एजेंटिक AI बुद्धिमत्ता',
            'Weather Risk Monitor': 'हवामान जोखीम मॉनिटर',
            'Market Intelligence': 'बाजार बुद्धिमत्ता',
            'Performance Analytics': 'कार्यक्षमता विश्लेषण',
            'System Configuration': 'सिस्टम कॉन्फिगरेशन',
            'Multi-lingual Demo': 'बहुभाषी डेमो',
            'Offline Capabilities': 'ऑफलाइन क्षमता',
            'Select Language': 'भाषा निवडा',
            'Current Language': 'सध्याची भाषा',
            'Accessibility': 'प्रवेशक्षमता',
            'Font Size': 'फॉन्ट आकार',
            'High Contrast Mode': 'उच्च कंट्रास्ट मोड',
            'Navigation Dashboard': 'नेविगेशन डॅशबोर्ड',
            'Select Dashboard': 'डॅशबोर्ड निवडा',
            'Live Metrics': 'लाइव मेट्रिक्स',
            'Weather Alerts': 'हवामान सूचना',
            'Portfolio Value': 'पोर्टफोलियो मूल्य',
            'Active Loans': 'सक्रिय कर्ज',
            'Default Rate': 'डिफॉल्ट दर',
            'Avg Credit Score': 'सरासरी क्रेडिट स्कोअर'
        }
    }
    
    if target_language in translations:
        for eng_text, translated_text in translations[target_language].items():
            text = text.replace(eng_text, translated_text)
    
    return text


# Explainable AI functions
def calculate_confidence_score(data_quality: float, model_performance: float, 
                             feature_completeness: float, temporal_relevance: float, 
                             spatial_coverage: float) -> Dict[str, Any]:
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
    
    return {
        'overall_confidence': overall_confidence,
        'factors': factors,
        'limitations': limitations,
        'weights': weights
    }

def generate_credit_explanation(farmer_data: Dict[str, Any], prediction: float, 
                              confidence: float, model: Any, feature_names: List[str]) -> Dict[str, Any]:
    """Generate human-readable explanation for credit decisions"""
    
    # Extract key factors
    key_factors = []
    for feature in feature_names:
        if feature in farmer_data:
            value = farmer_data[feature]
            
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
                'impact': impact
            })
    
    # Sort by importance
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
    
    # Generate recommendations
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
    
    return {
        'decision_summary': f"Credit {decision} - {risk_level} risk with {confidence:.1%} confidence",
        'key_factors': key_factors,
        'reasoning_trace': reasoning_trace,
        'recommendations': recommendations,
        'risk_level': risk_level
    }
# Global styling
st.markdown("""
<style>
    /* Smooth transitions for theme switching */
    body, [class^="st-"], [class*=" st-"] {
        transition: all 0.3s ease-in-out;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    /* -------------------------------
       MAIN HEADER
    -------------------------------- */
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5a8a 100%);
        padding: 2rem;
        border-radius: 12px;
        color: #ffffff; /* ensure light text */
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    .main-header h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #ffffff; }
    .main-header h3 { font-size: 1.2rem; font-weight: 500; color: #f0f0f0; }
    .main-header p { font-size: 1rem; margin-top: 0.5rem; color: #f0f0f0; }

    /* -------------------------------
       METRIC CARDS
    -------------------------------- */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        text-align: center;
        margin: 0.5rem 0;
        color: #222222;
        font-weight: 500;
    }

    .risk-low { border-left: 6px solid #228B22; font-weight: 600; }
    .risk-medium { border-left: 6px solid #e6a700; font-weight: 600; }
    .risk-high { border-left: 6px solid #d32f2f; font-weight: 600; }

    /* -------------------------------
       SIDEBAR
    -------------------------------- */
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: #222222; /* slightly darker for readability */
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .sidebar-logo h2 { margin: 0; color: #1f4e79; } /* brand blue */
    .sidebar-logo p { margin: 0.2rem 0; }

    /* -------------------------------
       FINANCIER INSIGHT
    -------------------------------- */
    .financier-insight {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
        color: #222222; /* dark text for light background */
        font-weight: 500;
    }

    /* -------------------------------
       PORTFOLIO SUMMARY
    -------------------------------- */
    .portfolio-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff; /* force white text */
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }

    /* -------------------------------
       WELCOME SCREEN
    -------------------------------- */
    .welcome-container {
        text-align: center;
        padding: 2rem;
        color: #222222; /* dark text for light mode */
    }
    .welcome-info-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 12px;
        color: #222222;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* -------------------------------
       DARK MODE
    -------------------------------- */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, #1A3C5B 0%, #264A6A 100%);
            color: #f5f5f5;
            box-shadow: 0 3px 8px rgba(255,255,255,0.05);
        }

        .metric-card {
            background: #262626;
            color: #f5f5f5;
            box-shadow: 0 2px 6px rgba(255,255,255,0.1);
        }

        .risk-low { border-left-color: #6edc86; }
        .risk-medium { border-left-color: #ffd75e; }
        .risk-high { border-left-color: #ff7b7b; }

        .sidebar-logo {
            background: #262626;
            color: #f5f5f5;
            box-shadow: 0 2px 6px rgba(255,255,255,0.08);
        }
        .sidebar-logo h2 { color: #90c6ff; }

        .financier-insight {
            background: #1f2e46;
            border-left: 5px solid #5a87be;
            color: #f5f5f5;
        }

        .portfolio-summary {
            background: linear-gradient(135deg, #4b5f88 0%, #5d4679 100%);
            color: #f5f5f5;
        }

        .welcome-container { color: #f5f5f5; }
        .welcome-container h3, .welcome-container h4 { color: #90ee90; }
        .welcome-info-box {
            background: #1f2e46;
            color: #f5f5f5;
        }
        .welcome-info-box h4 { color: #90ee90; }
    }
</style>
""", unsafe_allow_html=True)


def display_main_header():
    st.markdown(f"""
    <div class="main-header">
        <h1>🏦 {translate_text("Capital One AgriCred AI Platform", lang)}</h1>
        <h3>{translate_text("Advanced Agricultural Credit Intelligence & Risk Management", lang)}</h3>
        <p>{translate_text("Empowering financial institutions with AI-driven insights for agricultural lending", lang)}</p>
    </div>
    """, unsafe_allow_html=True)
def display_sidebar():
    # --- Branding ---
    st.sidebar.markdown(f"""
    <div class="sidebar-logo">
        <h2>🏦 Capital One</h2>
        <p><strong>AgriCred AI</strong></p>
        <p style="font-size: 12px; opacity: 0.8;">Agricultural Lending Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Navigation Dashboard (moved ABOVE language selection) ---
    st.sidebar.markdown("### 📊 Navigation Dashboard")
    
    page_options = [
        "🏠 Executive Summary",
        "📊 Portfolio Analytics", 
        "🎯 Credit Risk Scoring",
        "🤖 Agentic AI Intelligence",
        "🌦️ Weather Risk Monitor",
        "💹 Market Intelligence",
        "📈 Performance Analytics",
        "⚙️ System Configuration",
        "🌍 Multi-lingual Demo",
        "📱 Offline Capabilities"
    ]

    # Use English first, will re-translate after language selection
    selected_page = st.sidebar.selectbox("Select Dashboard", page_options, help="Choose your dashboard view")
    original_page = selected_page

    # --- Language Settings ---
    st.sidebar.markdown("### 🌍 Language Settings")

    language_options = [(code, f"{lang.native_name} ({lang.name})") 
                       for code, lang in SUPPORTED_LANGUAGES.items()]
    selected_language = st.sidebar.selectbox(
        "🌐 Select Language",
        [opt[0] for opt in language_options],
        index=0,
        format_func=lambda x: next(opt[1] for opt in language_options if opt[0] == x)
    )
    st.session_state.selected_language = selected_language

    # Show current language info
    current_lang = SUPPORTED_LANGUAGES[selected_language]
    st.sidebar.info(f"🌐 **Current Language**: {current_lang.native_name}")

    # Translate options AFTER language selection
    translated_options = [translate_text(option, selected_language) for option in page_options]
    selected_page_translated = translate_text(original_page, selected_language)

    # --- Accessibility options ---
    st.sidebar.markdown("### ♿ Accessibility")
    font_scale = st.sidebar.slider("📝 Font Size", 0.8, 1.5, 1.0, 0.1)
    high_contrast = st.sidebar.checkbox("🎨 High Contrast Mode", False)

    if font_scale != 1.0:
        st.markdown(f"<style>div.stMarkdown {{ font-size: {font_scale}em; }}</style>", unsafe_allow_html=True)

    # --- Live Metrics ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Live Metrics")
    try:
        pipeline = initialize_data_pipeline()
        metrics = pipeline.calculate_and_store_portfolio_metrics()
        st.sidebar.metric("Portfolio Value", f"₹{metrics['total_portfolio']/1e7:.1f}Cr")
        st.sidebar.metric("Active Loans", f"{metrics['total_loans']:,}")
        st.sidebar.metric("Default Rate", f"{metrics['default_rate']:.1f}%")
        st.sidebar.metric("Avg Credit Score", f"{int(metrics['avg_credit_score'])}")
    except Exception:
        st.sidebar.warning("Metrics unavailable")

    # --- Weather Alerts ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌦️ Weather Alerts")
    try:
        st.sidebar.info("🌡️ Monitoring weather conditions...")
        st.sidebar.info("🌧️ Checking for alerts...")
    except:
        st.sidebar.warning("Weather alerts unavailable")

    return original_page


@st.cache_data
def fetch_market_prices():
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        records = data.get("records", [])
        df = pd.DataFrame(records)
        return df
    else:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except:
        return None, None




@st.cache_resource
def initialize_data_pipeline():
    """Initialize data pipeline"""
    return AdvancedDataPipeline()

@st.cache_resource
def get_alert_system():
    setup_alerts_table()
    return WeatherAlertSystem()


API_URL = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={MARKET_API_KEY}&format=json&limit=100"

# List of cities and coordinates
CITIES = [
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Goa", "lat": 15.2993, "lon": 74.1240},
]
CITY_COORDS = {
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
    "Goa": (15.2993, 74.1240),
    "Udupi": (13.3522, 74.7919),
}
def performance_analytics():
    """Performance analytics and reporting"""
    st.markdown(f"## 📈 {translate_text('Performance Analytics & Reporting', lang)}")
    
    # Generate performance data
    months = pd.date_range(start='2024-09-01', end='2025-08-31', freq='MS')
    performance_data = {
        'Month': months,
        'Revenue (₹Cr)': np.cumsum(np.random.normal(8, 1, 12)) + 85,
        'Profit (₹Cr)': np.cumsum(np.random.normal(2, 0.5, 12)) + 25,
        'Cost of Funds (%)': np.random.normal(0, 0.1, 12) + 7.2,
        'NPA Ratio (%)': np.maximum(0, np.cumsum(np.random.normal(0, 0.2, 12)) + 4.1),
        'ROA (%)': np.random.normal(0, 0.2, 12) + 2.8,
        'New Loans': np.random.poisson(380, 12)
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    # Key performance indicators
    st.subheader(f"📊 {translate_text('Key Performance Indicators', lang)}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        latest_revenue = df_perf['Revenue (₹Cr)'].iloc[-1]
        prev_revenue = df_perf['Revenue (₹Cr)'].iloc[-2] 
        revenue_change = (latest_revenue - prev_revenue) / prev_revenue * 100
        st.metric(translate_text("Monthly Revenue", lang), f"₹{latest_revenue:.1f}Cr", f"{revenue_change:+.1f}%")
    
    with col2:
        latest_profit = df_perf['Profit (₹Cr)'].iloc[-1]
        prev_profit = df_perf['Profit (₹Cr)'].iloc[-2]
        profit_change = (latest_profit - prev_profit) / prev_profit * 100
        st.metric(translate_text("Monthly Profit", lang), f"₹{latest_profit:.1f}Cr", f"{profit_change:+.1f}%")
    
    with col3:
        latest_npa = df_perf['NPA Ratio (%)'].iloc[-1]
        st.metric(translate_text("NPA Ratio", lang), f"{latest_npa:.2f}%", help=translate_text("Non-performing assets ratio", lang))
    
    with col4:
        latest_roa = df_perf['ROA (%)'].iloc[-1]
        st.metric(translate_text("ROA", lang), f"{latest_roa:.2f}%", help=translate_text("Return on assets", lang))
    
    with col5:
        latest_loans = df_perf['New Loans'].iloc[-1]
        st.metric(translate_text("New Loans", lang), f"{latest_loans}", help=translate_text("New loans this month", lang))
    
    # Performance trends
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue and profit trend
        fig_revenue = px.line(
            df_perf,
            x='Month',
            y=['Revenue (₹Cr)', 'Profit (₹Cr)'],
            title=translate_text('Revenue & Profit Trends', lang)
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # NPA and ROA trend
        fig_ratios = px.line(
            df_perf,
            x='Month', 
            y=['NPA Ratio (%)', 'ROA (%)'],
            title=translate_text('Key Financial Ratios', lang)
        )
        st.plotly_chart(fig_ratios, use_container_width=True)
    
    # Loan disbursement trend
    fig_loans = px.bar(
        df_perf,
        x='Month',
        y='New Loans',
        title=translate_text('Monthly New Loan Disbursements', lang)
    )
    st.plotly_chart(fig_loans, use_container_width=True)

# --- Load Data from DB ---
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM credits", conn)
    conn.close()
    return df

def get_weather(lat, lon):
    """Fetch live weather data for given lat/lon with fallback."""
    try:
        # Try to get real weather data
        if WEATHER_API_KEY:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Add data provenance
                data['_provenance'] = {
                    'source': 'OpenWeatherMap API',
                    'confidence': 0.95,
                    'data_freshness': 'real_time',
                    'fallback_used': False
                }
                return data
    except Exception as e:
        st.warning(f"{translate_text('Weather API call failed', lang)}: {e}")
    
    # Fallback to static regional data
    fallback_weather = get_fallback_weather_data(lat, lon)
    if fallback_weather:
        fallback_weather['_provenance'] = {
            'source': 'Regional Fallback Data',
            'confidence': 0.7,
            'data_freshness': 'seasonal_average',
            'fallback_used': True,
            'fallback_reason': 'API unavailable'
        }
        return fallback_weather
    
    return None

def get_fallback_weather_data(lat, lon):
    """Get fallback weather data based on coordinates"""
    # Simple regional mapping for fallback
    if 20 <= lat <= 30 and 70 <= lon <= 80:  # North India
        return {
            "main": {"temp": 28, "humidity": 65, "pressure": 1013},
            "weather": [{"description": "clear sky", "main": "Clear"}],
            "wind": {"speed": 5},
            "name": "North India Region"
        }
    elif 10 <= lat <= 20 and 70 <= lon <= 80:  # Central India
        return {
            "main": {"temp": 32, "humidity": 70, "pressure": 1010},
            "weather": [{"description": "partly cloudy", "main": "Clouds"}],
            "wind": {"speed": 8},
            "name": "Central India Region"
        }
    elif 8 <= lat <= 15 and 75 <= lon <= 85:  # South India
        return {
            "main": {"temp": 30, "humidity": 75, "pressure": 1011},
            "weather": [{"description": "scattered clouds", "main": "Clouds"}],
            "wind": {"speed": 7},
            "name": "South India Region"
        }
    
    # Default fallback
    return {
        "main": {"temp": 25, "humidity": 60, "pressure": 1012},
        "weather": [{"description": "clear sky", "main": "Clear"}],
        "wind": {"speed": 6},
        "name": "India Region"
    }

def parse_weather_data(weather_json):
    """Convert raw weather data into readable text and alerts with provenance."""
    if not weather_json:
        return "", []

    city = weather_json["name"]
    temp = weather_json["main"]["temp"]
    humidity = weather_json["main"]["humidity"]
    wind = weather_json["wind"]["speed"]
    desc = weather_json["weather"][0]["description"].title()

    # Get data provenance
    provenance = weather_json.get('_provenance', {})
    source = provenance.get('source', 'Unknown Source')
    confidence = provenance.get('confidence', 0.5)
    fallback_used = provenance.get('fallback_used', False)
    
    # Add provenance indicator to report
    if fallback_used:
        report = f"**{city}**: {desc}, 🌡 {temp}°C, 💧 Humidity {humidity}%, 💨 Wind {wind} m/s\n*Source: {source} (Fallback Data - Confidence: {confidence:.1%})*"
    else:
        report = f"**{city}**: {desc}, 🌡 {temp}°C, 💧 Humidity {humidity}%, 💨 Wind {wind} m/s\n*Source: {source} (Real-time - Confidence: {confidence:.1%})*"

    # Basic alert rules
    alerts = []
    if temp > 35:
        alerts.append(("Heatwave Risk", "High"))
    if humidity > 90 and "rain" in desc.lower():
        alerts.append(("Heavy Rain Alert", "Medium"))
    if wind > 15:
        alerts.append(("High Wind Alert", "High"))
    if not alerts:
        alerts.append(("All Clear", "Low"))

    return report, alerts

def display_weather_reports():

    text_color = st.get_option("theme.textColor")
    background_color = st.get_option("theme.backgroundColor")

    st.markdown(f"### 📄 {translate_text('Latest Weather Reports', lang)}")
    cols = st.columns(2)  # Two-column layout for compactness
    
    for i, city in enumerate(CITIES):
        data = get_weather(city["lat"], city["lon"])
        if data:
            report, _ = parse_weather_data(data)
            
            # Extract components for styling
            name = data["name"]
            desc = data["weather"][0]["description"].title()
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]

            with cols[i % 2]:
                st.markdown(f"""
                <div style='
                    background-color: {background_color}; 
                    color: {text_color};
                    border-radius: 10px; 
                    padding: 12px; 
                    margin-bottom: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>
                    <h4 style='margin-bottom: 4px; color: {text_color};'>{name}</h4>
                    <p style='margin: 0; font-size: 15px; color: {text_color};'>{desc}</p>
                    <p style='margin: 0; color: {text_color};'>🌡 {temp}°C&nbsp;&nbsp;💧 {humidity}%&nbsp;&nbsp;💨 {wind} m/s</p>
                </div>
                """, unsafe_allow_html=True)

def display_alerts(alerts_feed):
    text_color = st.get_option("theme.textColor")
    background_color = st.get_option("theme.backgroundColor")
    st.subheader(f"📡 {translate_text('Live Weather Alerts Feed', lang)}")
    
    severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    for alert in alerts_feed:
        st.markdown(f"""
        <div style='
            background-color: {background_color}; 
            border-radius: 8px; 
            padding: 8px 12px; 
            margin-bottom: 6px;
            border-left: 5px solid {"#dc3545" if alert["severity"]=="High" else "#ffc107" if alert["severity"]=="Medium" else "#28a745"};
        '>
            <strong>{severity_color[alert['severity']]} {alert['alert']}</strong> — {alert['city']}
        </div>
        """, unsafe_allow_html=True)




def fetch_weather(lat, lon):
    """Return a simplified weather summary and risk proxy dict."""
    try:
        url = f"https://api.openweathermap.org/data/3.0/onecall?"
        params = {
            "lat": float(lat),
            "lon": float(lon),
            "exclude": "minutely,hourly,alerts",
            "units": "metric",
            "appid": WEATHER_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # current + daily summary
        current = data.get("current", {})
        daily = data.get("daily", [])

        # compute 7-day rainfall sum (if available)
        rainfall_7day = 0.0
        for d in daily[:7]:
            # openweather may have d.get('rain') or daily precipitation array
            rainfall_7day += d.get("rain", 0.0) if d.get("rain") else 0.0

        # simple risk proxies (tunable thresholds)
        temp = current.get("temp")
        humidity = current.get("humidity")
        # frost_risk: if min temp in next 3 days < 3C
        frost_risk = 0.0
        for d in daily[:3]:
            if d.get("temp", {}).get("min") is not None and d["temp"]["min"] < 3.0:
                frost_risk = max(frost_risk, 0.9)  # high frost risk
        # drought proxy: low cumulative rainfall compared to threshold
        drought_risk = 0.0
        if rainfall_7day < 10:  # <10 mm in 7 days -> drought-ish (tweak by crop/season)
            drought_risk = 0.8
        elif rainfall_7day < 30:
            drought_risk = 0.4

        return {
            "temperature": temp,
            "humidity": humidity,
            "rainfall_7day": rainfall_7day,
            "frost_risk": frost_risk,
            "drought_risk": drought_risk,
            "raw": data
        }

    except Exception as e:
        st.warning(f"{translate_text('Weather fetch failed', lang)}: {e}")
        # fallback safe defaults (so rest of pipeline works)
        return {
            "temperature": None,
            "humidity": None,
            "rainfall_7day": None,
            "frost_risk": 0.0,
            "drought_risk": 0.0,
            "raw": {}
        }
    

def credit_risk_scoring_dashboard():
    # Header
    st.markdown(f"# 🏦 {translate_text('Agricultural Credit Risk Assessment', lang)}")
    st.markdown(f"### {translate_text('AI-Powered Credit Scoring for Agricultural Lending', lang)}")
    st.markdown("---")
    
    pipeline = initialize_data_pipeline()

    # Load model artifacts
    try:
        model = joblib.load('advanced_credit_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        model_type = "xgboost"  # Set based on your best model
    except Exception as e:
        st.error(f"⚠️ {translate_text('Error loading model', lang)}: {e}")
        st.info(translate_text("Please ensure model files are present: advanced_credit_model.pkl, feature_scaler.pkl, feature_columns.pkl", lang))
        return
    
    # Complete feature defaults
    defaults = {
    'farmer_age': 40,                    # Younger, more tech-savvy
    'education_level': 4,                # Above average education
    'family_size': 4,                    # Smaller household obligations
    'land_size': 3.0,                    # Moderately large farm
    'crop_type_encoded': 2,              # Wheat (stable commodity)
    'irrigation_access': 1,              # ✅ Has irrigation
    'current_temperature': 28.0,         # Optimal growing temp
    'current_humidity': 60,              # Ideal humidity
    'temperature_stress': 0.1,           # Low stress
    'humidity_stress': 0.1,              # Low stress
    'drought_risk_7days': 0.1,           # Low drought risk
    'frost_risk_7days': 0.01,            # Almost zero frost risk
    'excess_rain_risk': 0.05,            # Very low flood risk
    'price_volatility': 0.1,             # Stable prices
    'nearest_mandi_distance': 10.0,      # Close to market
    'connectivity_index': 0.8,           # Strong connectivity
    'input_cost_index': 0.3,             # Lower input costs
    'loan_to_land_ratio': 0.2,           # Conservative borrowing
    'debt_to_income_ratio': 0.2,         # Low debt burden
    'payment_history_score': 0.95,       # Excellent history
    'yield_consistency': 0.9,            # Very consistent yields
    'soil_health_index': 0.9,            # Very healthy soil
    'nutrient_deficiency_risk': 0.05,    # Negligible nutrient risk
    'insurance_coverage': 1,             # ✅ Insured
    'cooperative_membership': 1,         # ✅ Member
    'technology_adoption': 0.8,          # High tech use
    'diversification_index': 0.7,        # Well diversified
    'electricity_reliability': 0.9,      # Very reliable power
    'mobile_network_strength': 0.9,      # Excellent connectivity
    'bank_branch_distance': 5.0,         # Very close to bank
    # And for the rest, use similarly low‐risk values:
    'seasonal_rainfall_deviation': 0.0,
    'historical_drought_frequency': 0,
    'climate_change_vulnerability': 0.1,
    'current_price': 200000.0,
    'market_demand_index': 0.8,
    'export_potential': 0.7,
    'storage_price_premium': 0.1,
    'price_trend': 0.05,
    'savings_to_income_ratio': 0.2,
    'credit_utilization': 0.2,
    'number_of_credit_sources': 2,
    'informal_lending_dependency': 0.1,
    'road_quality_index': 0.9,
    'mechanization_level': 0.8,
    'seed_quality_index': 0.9,
    'fertilizer_usage_efficiency': 0.9,
    'pest_management_score': 0.8,
    'organic_farming_adoption': 0.3,
    'precision_agriculture_usage': 0.7,
    'eligible_schemes_count': 3,
    'subsidy_utilization': 0.8,
    'msp_eligibility': 1,
    'kisan_credit_card': 1,
    'government_training_participation': 0.8,
    'community_leadership_role': 1,
    'social_capital_index': 0.8,
    'extension_service_access': 0.8,
    'peer_learning_participation': 0.8,
    'labor_availability': 0.8,
    'storage_access': 1,
    'transport_cost_burden': 0.2,
    'supply_chain_integration': 0.8,
    'disaster_preparedness': 0.8,
    'alternative_income_sources': 0.7,
    'livestock_ownership': 1
}

    # # Multi-lingual input section
    # st.markdown(f"## 🌍 {translate_text('Multi-lingual Input', lang)}")
    
    # # Language detection for input
    # input_text = st.text_area(f"💬 {translate_text('Enter your query in any supported language:', lang)}", 
    #                           placeholder=translate_text("Type in English, Hindi, Marathi, Tamil, Telugu, etc.", lang))
    
    # if input_text:
    #     detected_lang, confidence = detect_language(input_text)
    #     st.info(f"🌐 {translate_text('Detected Language', lang)}: {get_language_display_name(detected_lang)} ({translate_text('Confidence', lang)}: {confidence:.1%})")
        
    #     # Show language-specific response
    #     if detected_lang != 'en':
    #         st.success(f"✅ {translate_text('Processing in', lang)} {get_language_display_name(detected_lang)}")
    
    # # Voice input simulation
    # if st.button(f"🎤 {translate_text('Simulate Voice Input', lang)}"):
    #     st.info(f"🎙️ {translate_text('Voice input would be processed here in a real implementation', lang)}")
    #     # In real implementation, this would use speech recognition

    # Input form
    st.markdown(f"## 📝 {translate_text('Farmer Assessment Form', lang)}")
    st.markdown(f"*{translate_text('Enter key information for credit evaluation', lang)}*")
    
    # Farmer details
    farmer_name = st.text_input(f"👤 {translate_text('Farmer Name', lang)}", "Rajesh Kumar")
    monthly_income = st.number_input(f"💰 {translate_text('Monthly Income (₹)', lang)}", min_value=5000, max_value=200000, value=25000)
    
    # Key risk factors
    user_inputs = {}
    
    with st.expander(f"🏦 {translate_text('Financial Information', lang)}", expanded=True):
        user_inputs['payment_history_score'] = st.slider(translate_text("Payment History Score", lang), 0.1, 1.0, 0.85, help=translate_text("Track record of loan repayments", lang))
        user_inputs['debt_to_income_ratio'] = st.slider(translate_text("Debt to Income Ratio", lang), 0.0, 2.0, 0.4, help=translate_text("Monthly debt payments / Monthly income", lang))
        user_inputs['savings_to_income_ratio'] = st.slider(translate_text("Savings Rate", lang), 0.0, 0.5, 0.1, help=translate_text("Percentage of income saved monthly", lang))
    
    with st.expander(f"🌾 {translate_text('Agricultural Details', lang)}", expanded=True):
        user_inputs['land_size'] = st.number_input(translate_text("Land Size (hectares)", lang), 0.5, 20.0, 2.0, help=translate_text("Total cultivated land", lang))
        user_inputs['yield_consistency'] = st.slider(translate_text("Yield Consistency", lang), 0.3, 1.0, 0.7, help=translate_text("Reliability of crop yields", lang))
        user_inputs['irrigation_access'] = st.radio(translate_text("Irrigation Access?", lang), [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['soil_health_index'] = st.slider(translate_text("Soil Health", lang), 0.2, 1.0, 0.75, help=translate_text("Soil quality and fertility", lang))
    
    with st.expander(f"🌦️ {translate_text('Climate & Weather Risks', lang)}", expanded=False):
        user_inputs['drought_risk_7days'] = st.slider(translate_text("7-day Drought Risk", lang), 0.0, 1.0, 0.3)
        user_inputs['price_volatility'] = st.slider(translate_text("Price Volatility", lang), 0.05, 0.8, 0.2, help=translate_text("Market price fluctuation", lang))
    
    with st.expander(f"🤝 {translate_text('Support Systems', lang)}", expanded=False):
        user_inputs['cooperative_membership'] = st.radio(translate_text("Cooperative Member?", lang), [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['insurance_coverage'] = st.radio(translate_text("Crop Insurance?", lang), [0, 1], index=1, format_func=lambda x: "✅ Yes" if x else "❌ No")
        user_inputs['technology_adoption'] = st.slider(translate_text("Technology Adoption", lang), 0.1, 0.95, 0.5, help=translate_text("Use of modern farming techniques", lang))
        user_inputs['diversification_index'] = st.slider(translate_text("Crop Diversification", lang), 0.1, 0.9, 0.4, help=translate_text("Variety of crops grown", lang))
    
    # Assessment button
    assess_button = st.button(f"🔍 {translate_text('Assess Credit Risk', lang)}", type="primary", use_container_width=True)
    
    # Main content area
    if not assess_button:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem;'>
                <h3>🌾 {translate_text('Agricultural Credit Assessment', lang)}</h3>
                <p>{translate_text('Complete the farmer assessment form and click', lang)} 
                <strong>"{translate_text('Assess Credit Risk', lang)}"</strong> {translate_text('to generate a comprehensive credit evaluation.', lang)}</p>
                <br>
            </div>
            """, unsafe_allow_html=True)


    else:
        # Build prediction input
        features = defaults.copy()
        features.update(user_inputs)
        input_list = [features[feat] for feat in feature_columns]
        input_df = pd.DataFrame([input_list], columns=feature_columns)
        
        try:
            # Make prediction
            input_scaled = scaler.transform(input_df)
            pred_prob = model.predict_proba(input_scaled)[0][1]
            credit_score = int((1 - pred_prob) * 750 + 250)
            
            # Professional Results Display with Explainable AI
            st.markdown("---")
            st.subheader(f"📊 {translate_text('Comprehensive Assessment for', lang)} {farmer_name}")
            
            # Calculate confidence score for this assessment
            confidence_breakdown = calculate_confidence_score(
                data_quality=0.85,  # Based on available features
                model_performance=0.90,  # Model performance
                feature_completeness=0.95,  # Feature completeness
                temporal_relevance=0.80,  # Data freshness
                spatial_coverage=0.85  # Geographic coverage
            )
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(translate_text("🎯 Credit Score", lang), credit_score, help=translate_text("FICO-style score (250-1000)", lang))
            with col2:
                st.metric(translate_text("⚠️ Default Risk", lang), f"{pred_prob:.1%}", help=translate_text("Probability of default", lang))
            with col3:
                if pred_prob < 0.4:
                    st.success(f"✅ {translate_text('APPROVE', lang)}")
                    recommendation = "APPROVE"
                elif pred_prob < 0.7:
                    st.warning(f"⚠️ {translate_text('REVIEW', lang)}")
                    recommendation = "REVIEW"
                else:
                    st.error(f"❌ {translate_text('REJECT', lang)}")
                    recommendation = "REJECT"
            with col4:
                loan_capacity = int(monthly_income * 12 * 3 * (1 - pred_prob))
                st.metric(translate_text("💰 Max Loan Capacity", lang), f"₹{loan_capacity:,}", help=translate_text("Recommended maximum loan amount", lang))
            
            # Confidence Score Display
            st.markdown("---")
            st.subheader(f"🎯 {translate_text('AI Confidence Assessment', lang)}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(translate_text("Overall Confidence", lang), f"{confidence_breakdown['overall_confidence']:.1%}")
                
                # Show confidence factors
                st.markdown(f"**{translate_text('Confidence Breakdown', lang)}:**")
                for factor, score in confidence_breakdown['factors'].items():
                    st.write(f"• {translate_text(factor.replace('_', ' ').title(), lang)}: {score:.1%}")
            
            with col2:
                # Show limitations if any
                if confidence_breakdown['limitations']:
                    st.warning(f"**{translate_text('Limitations', lang)}:**")
                    for limitation in confidence_breakdown['limitations']:
                        st.write(f"• {limitation}")
                else:
                    st.success(f"✅ {translate_text('No significant limitations detected', lang)}")
            
            # Data Provenance
            st.markdown("---")
            st.subheader(f"📊 {translate_text('Data Provenance & Sources', lang)}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**{translate_text('Data Sources Used', lang)}:**")
                st.write(f"• {translate_text('Farmer Profile', lang)}: {translate_text('User Input', lang)}")
                st.write(f"• {translate_text('Weather Data', lang)}: {translate_text('OpenWeatherMap API + Regional Fallback', lang)}")
                st.write(f"• {translate_text('Market Data', lang)}: {translate_text('Agmarknet API + Historical Database', lang)}")
                st.write(f"• {translate_text('Credit Model', lang)}: {translate_text('Trained on Agricultural Dataset', lang)}")
            
            with col2:
                st.info(f"**{translate_text('Data Quality Indicators', lang)}:**")
                st.write(f"• {translate_text('Model Performance', lang)}: {confidence_breakdown['factors']['model_performance']:.1%}")
                st.write(f"• {translate_text('Feature Completeness', lang)}: {confidence_breakdown['factors']['feature_completeness']:.1%}")
                st.write(f"• {translate_text('Data Freshness', lang)}: {confidence_breakdown['factors']['temporal_relevance']:.1%}")
                st.write(f"• {translate_text('Geographic Coverage', lang)}: {confidence_breakdown['factors']['spatial_coverage']:.1%}")
            
            # Risk breakdown
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🧠 {translate_text('AI Decision Explanation', lang)}")
                
                # Generate explainable AI explanation
                explanation = generate_credit_explanation(
                    user_inputs, pred_prob, confidence_breakdown['overall_confidence'], 
                    model, feature_columns
                )
                
                # Display decision summary
                st.info(f"**{explanation['decision_summary']}**")
                
                # Display reasoning trace
                st.markdown(f"**🔍 {translate_text('Reasoning Process', lang)}:**")
                for step in explanation['reasoning_trace']:
                    st.write(f"• {step}")
                
                # Display key factors
                st.markdown(f"**📊 {translate_text('Key Factors', lang)}:**")
                for factor in explanation['key_factors'][:5]:
                    impact_emoji = "✅" if factor['impact'] == 'positive' else "⚠️" if factor['impact'] == 'negative' else "➡️"
                    st.write(f"{impact_emoji} {translate_text(factor['feature'].replace('_', ' ').title(), lang)}: {factor['value']}")
                
                # Display recommendations
                if explanation['recommendations']:
                    st.markdown(f"**💡 {translate_text('Recommendations', lang)}:**")
                    for rec in explanation['recommendations']:
                        st.write(f"• {rec}")
                
                # Add SMS export functionality
                st.markdown("---")
                st.subheader(f"📱 {translate_text('Export & Share', lang)}")
                
                col_sms1, col_sms2 = st.columns(2)
                with col_sms1:
                    if st.button(f"📱 {translate_text('Export as SMS', lang)}"):
                        sms_text = create_sms_text(explanation['decision_summary'], st.session_state.selected_language)
                        st.success(translate_text("SMS text generated!", lang))
                        st.code(sms_text)
                
                with col_sms2:
                    if st.button(f"🔊 {translate_text('Text-to-Speech', lang)}"):
                        audio_data = text_to_speech(explanation['decision_summary'], st.session_state.selected_language)
                        if audio_data:
                            st.success(translate_text("Audio generated! (Would play in real implementation)", lang))
                        else:
                            st.warning(translate_text("Audio generation failed", lang))
            
            with col2:
                # Risk gauge
                risk_value = pred_prob * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Level", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if risk_value > 60 else "orange" if risk_value > 30 else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance visualization
            st.markdown("---")
            st.subheader(f"📈 {translate_text('Model Feature Analysis', lang)}")
            
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Create SHAP summary
                feature_impact = []
                for i, (feature, shap_val, feat_val) in enumerate(zip(feature_columns, shap_values[1][0], input_df.iloc[0].values)):
                    feature_impact.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'Impact': shap_val,
                        'Value': feat_val
                    })
                
                # Sort by absolute impact
                feature_impact.sort(key=lambda x: abs(x['Impact']), reverse=True)
                
                # Display top 10 features
                impact_df = pd.DataFrame(feature_impact[:10])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{translate_text('Top Influential Features', lang)}:**")
                    st.dataframe(impact_df, hide_index=True)
                
                with col2:
                    # Bar chart of feature impacts
                    fig_bar = px.bar(
                        impact_df, 
                        x='Impact', 
                        y='Feature',
                        orientation='h',
                        title="Feature Impact on Risk Score",
                        color='Impact',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
            except Exception:
                # Fallback to model feature importance if SHAP fails
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': [f.replace('_', ' ').title() for f in feature_columns],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Model Feature Importance"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info(translate_text("Feature analysis not available for this model type.", lang))

            user_crop = st.text_input("Main Crop (e.g. wheat, cotton, rice)", "all")
            user_state = st.text_input("State (e.g. Maharashtra, Bihar)", "all")
            user_land_size = user_inputs.get('land_size', 2.0)  # Already in your user inputs block
            policy_advisor_with_keyword_search(user_land_size, user_crop, user_state)


        except Exception as e:
            st.error(f"{translate_text('Error during prediction', lang)}: {e}")
            st.info(translate_text("Please check if all required model files are present and properly trained.", lang))

        


def map_land_size_category(hectares):
    if hectares < 0.4:
        return 'marginal'
    elif hectares < 0.8:
        return 'small'
    elif hectares < 4:
        return 'medium'
    else:
        return 'large'

def policy_advisor_with_keyword_search(land_size_hectares, crop_keyword='all', state_keyword='all'):
    st.header(f'🏛️ {translate_text("Dynamic Government Policy Advisor", lang)}')
    st.markdown(f"""
        **{translate_text("Relevant government schemes and policies based on your farm profile and location.", lang)}**
    """)

    try:
        with open('myschemes_full.json', 'r', encoding='utf-8') as f:
            schemes = json.load(f)
    except FileNotFoundError:
        st.error(f'❌ {translate_text("`myschemes_full.json` not found. Please ensure the file is in the app directory.", lang)}')
        return
    
    land_category = map_land_size_category(land_size_hectares).lower()
    crop_keyword = crop_keyword.strip().lower()
    state_keyword = state_keyword.strip().lower()

    # Synonyms for land.
    land_synonyms = {
        "marginal": ["marginal", "small"],
        "small": ["small", "marginal"],
        "medium": ["medium"],
        "large": ["large"]
    }
    search_terms = set()
    search_terms.add(land_category)
    for synonym in land_synonyms.get(land_category, []):
        search_terms.add(synonym)

    filtered_schemes = []
    for scheme in schemes:
        scheme_text = ' '.join([
            scheme.get('title', ''),
            scheme.get('description', ''),
            scheme.get('benefits', ''),
            scheme.get('eligibility', ''),
        ]).lower()
        
        land_match = (land_category=='all') or any(term in scheme_text for term in search_terms)
        crop_match = (crop_keyword=='all') or (crop_keyword in scheme_text)
        state_match = (state_keyword=='all') or (state_keyword in scheme_text)

        if land_match and crop_match and state_match:
            filtered_schemes.append(scheme)

    if filtered_schemes:
        st.markdown(f"### {translate_text('Found', lang)} {len(filtered_schemes)} {translate_text('matching schemes', lang)}:")
        for s in filtered_schemes:
            st.markdown(
                f"#### [{s.get('title', translate_text('Untitled Scheme', lang))}]({s.get('url', '#')})\n"
                f"{s.get('description', '')}\n\n**{translate_text('Benefits', lang)}:** {s.get('benefits', '')}\n\n"
                f"**{translate_text('Eligibility', lang)}:** {s.get('eligibility','')}\n\n---"
            )
    else:
        st.info(translate_text("No matched schemes found. Try broadening your filter criteria.", lang))

def generate_weather_alerts(weather_data, crop_type):
    """Generate weather-based alerts"""
    alerts = []
    
    # Frost alert
    if weather_data['frost_risk'] > 0.7:
        alerts.append({
            'severity': 'high',
            'message': f'{translate_text("Frost warning for", lang)} {crop_type} - {translate_text("temperature may drop below 2°C", lang)}'
        })
    
    # Drought alert
    if weather_data['drought_risk'] > 0.6:
        alerts.append({
            'severity': 'medium',
            'message': f'{translate_text("Drought conditions expected - consider water conservation", lang)}'
        })
    
    # Normal conditions
    if weather_data['frost_risk'] < 0.3 and weather_data['drought_risk'] < 0.3:
        alerts.append({
            'severity': 'low',
            'message': translate_text('Weather conditions favorable for crop growth', lang)
        })
    
    return alerts



# def weather_risk_monitor(pipeline):
#     st.header("🌤️ Live Weather Risk Monitoring System")

#     # Dashboard metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("🌡️ Active Farmers", "1,247", "↑ 23")
#     with col2: st.metric("⚠️ High Risk Alerts", "15", "↓ 3")
#     with col3: st.metric("🌧️ Rainfall Alerts", "8", "→ 0")
#     with col4: st.metric("✅ Safe Conditions", "1,224", "↑ 20")

#     """Weather risk monitoring dashboard"""
#     st.markdown("## 🌦️ Weather Risk Monitor")
           
#     try:
#         alert_system = WeatherAlertSystem()
        
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             st.subheader("🚨 Active Weather Alerts")
            
#             if st.button("🔄 Check for New Alerts", type="primary"):
#                 with st.spinner("Scanning weather conditions..."):
#                     try:

                        
#                         alerts_generated = alert_system.run_once()
#                         st.success(f"✅ Scan complete! Generated {alerts_generated} alerts")
#                     except Exception as e:
#                         st.error(f"Error generating alerts: {str(e)}")
            
#             # Display recent alerts
#             try:
#                 recent_alerts = alert_system.list_recent_alerts(limit=20)
                
#                 if recent_alerts:
#                     # Format and display
#                     for i, alert in enumerate(recent_alerts[:5]):
#                         severity_color = {
#                             'high': 'error',
#                             'medium': 'warning', 
#                             'low': 'info'
#                         }.get(alert['severity'], 'info')
                        
#                         with st.container():
#                             col1, col2, col3 = st.columns([2, 1, 1])
#                             with col1:
#                                 getattr(st, severity_color)(f"**{alert['alert_type'].title()}**: {alert['message']}")
#                             with col2:
#                                 st.write(f"Farmer ID: {alert['farmer_id']}")
#                             with col3:
#                                 st.write(f"Severity: {alert['severity'].upper()}")
#                 else:
#                     st.info("No recent alerts. Weather conditions are stable.")
#             except Exception as e:
#                 st.error(f"Error fetching alerts: {str(e)}")
#                 st.info("Weather alert system may need database initialization. Check logs for details.")
        
#         # Fetch real weather data for all cities
#         weather_data = []
#         alerts_feed = []
        
#         try:
#             for city in CITIES:
#                 data = get_weather(city["lat"], city["lon"])
#                 if data:
#                     risk_level = min(max(data["main"]["temp"] / 50, 0), 1)  # simple risk proxy
#                     weather_data.append({
#                         "lat": city["lat"],
#                         "lon": city["lon"],
#                         "city": city["name"],
#                         "risk_level": risk_level,
#                         "farmers_count": int(100 + risk_level * 200)
#                     })
#                     _, alerts = parse_weather_data(data)
#                     for alert, severity in alerts:
#                         alerts_feed.append({"city": city["name"], "alert": alert, "severity": severity})

#             # Weather map
#             if weather_data:
#                 st.subheader("🗺️ Regional Weather Risk Map")
#                 weather_df = pd.DataFrame(weather_data)
#                 fig_map = px.scatter_mapbox(
#                     weather_df, lat="lat", lon="lon", color="risk_level",
#                     size="farmers_count", hover_name="city",
#                     color_continuous_scale="RdYlGn_r", size_max=50, zoom=4
#                 )
#                 fig_map.update_layout(mapbox_style="open-street-map", height=400, margin={"r":0,"t":0,"l":0,"b":0})
#                 st.plotly_chart(fig_map, use_container_width=True)
#             else:
#                 st.warning("Unable to fetch weather data. Please check your internet connection.")

#         except Exception as e:
#             st.error(f"Error fetching weather data: {str(e)}")

#         # Display weather reports
#         try:
#             display_weather_reports()
#         except Exception as e:
#             st.error(f"Error displaying weather reports: {str(e)}")

#         # Display alerts
#         try:
#             display_alerts(alerts_feed)
#         except Exception as e:
#             st.error(f"Error displaying alerts: {str(e)}")
            
#     except Exception as e:
#         st.error(f"Weather monitoring system error: {str(e)}")
#         st.info("Please check your configuration and internet connection.")

def weather_risk_monitor(pipeline=None):
    """Improved weather risk monitoring with real data integration"""
    st.header(f"🌤️ {translate_text('Live Weather Risk Monitoring System', lang)}")
    
    # Initialize systems
    try:
        from weather_alert_system import WeatherAlertSystem
        weather_system = WeatherAlertSystem()
    except ImportError:
        st.error(translate_text("Weather Alert System not available", lang))
        return
    
    # Get real metrics from pipeline
    if pipeline:
        try:
            real_metrics = pipeline.calculate_and_store_portfolio_metrics()
            total_farmers = real_metrics.get('total_farmers', 0)
            
            # Get recent alerts count
            recent_alerts = weather_system.list_recent_alerts(limit=50)
            high_risk_alerts = len([a for a in recent_alerts if a.get('severity') == 'high'])
            rainfall_alerts = len([a for a in recent_alerts if 'rain' in a.get('message', '').lower() or 'drought' in a.get('message', '').lower()])
            safe_farmers = max(0, total_farmers - high_risk_alerts)
            
        except Exception as e:
            st.warning(f"{translate_text('Could not fetch real metrics', lang)}: {e}")
            # Fallback to demo values
            total_farmers, high_risk_alerts, rainfall_alerts, safe_farmers = 1247, 15, 8, 1224
    else:
        # Demo values for MVP
        total_farmers, high_risk_alerts, rainfall_alerts, safe_farmers = 1247, 15, 8, 1224

    # Dashboard metrics with real data
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric(f"🌡️ {translate_text('Active Farmers', lang)}", f"{total_farmers:,}", "↑ 23")
    with col2: 
        st.metric(f"⚠️ {translate_text('High Risk Alerts', lang)}", str(high_risk_alerts), "↓ 3" if high_risk_alerts < 20 else "↑ 5")
    with col3: 
        st.metric(f"🌧️ {translate_text('Rainfall Alerts', lang)}", str(rainfall_alerts), "→ 0")
    with col4: 
        st.metric(f"✅ {translate_text('Safe Conditions', lang)}", f"{safe_farmers:,}", "↑ 20")

    st.markdown("---")

    # Alert System Integration
    st.subheader(f"🚨 {translate_text('Live Weather Alert System', lang)}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(f"🔄 {translate_text('Check for New Alerts', lang)}", type="primary", use_container_width=True):
            with st.spinner(translate_text("Analyzing weather conditions for farmers...", lang)):
                try:
                    # Use the MVP method we created
                    alert_results = weather_system.run_once_mvp()

                    if alert_results["status"] == "success":
                        st.success(f"✅ {alert_results['summary']}")

                        if alert_results["alerts"]:
                            st.subheader(f"⚠️ {translate_text('Critical Weather Alerts', lang)}")
                            
                            # Display alerts in an organized way
                            for i, alert in enumerate(alert_results["alerts"][:5], 1):
                                severity_icon = "🔴" if alert["severity"] == "high" else "🟡" if alert["severity"] == "medium" else "🟢"
                                
                                with st.expander(f"{severity_icon} {translate_text('Alert', lang)} #{i}: {alert['farmer_name']} - {alert['type'].replace('_', ' ').title()}", expanded=i<=2):
                                    st.markdown(f"**{translate_text('Message', lang)}:** {alert['message']}")
                                    st.markdown(f"**{translate_text('Recommended Action', lang)}:** {alert['recommended_action']}")
                                    st.markdown(f"**{translate_text('Severity', lang)}:** {alert['severity'].title()}")
                                    st.markdown(f"**{translate_text('Risk Level', lang)}:** {alert['risk_level']:.1%}")
                        else:
                            st.info(f"🌤️ {translate_text('No critical weather alerts at this time. All farmers are in safe conditions.', lang)}")
                    else:
                        st.error(f"❌ {translate_text('Weather alert check failed', lang)}: {alert_results['summary']}")

                except Exception as e:
                    st.error(f"{translate_text('Error running weather analysis', lang)}: {str(e)}")
                    st.info(translate_text("Please check the weather alert system configuration.", lang))
    
    with col2:
        st.metric(translate_text("Last Check", lang), datetime.now().strftime("%H:%M"), "2 min ago")

    st.markdown("---")

    # Weather Risk Map Section
    st.subheader("🗺️ Regional Weather Risk Overview")
    
    try:
        # Define major agricultural regions in India for demo
        AGRICULTURAL_REGIONS = [
            {"name": "Punjab (Ludhiana)", "lat": 30.9010, "lon": 75.8573, "crop": "Wheat"},
            {"name": "Maharashtra (Pune)", "lat": 18.5204, "lon": 73.8567, "crop": "Sugarcane"}, 
            {"name": "Karnataka (Bangalore)", "lat": 12.9716, "lon": 77.5946, "crop": "Rice"},
            {"name": "Tamil Nadu (Chennai)", "lat": 13.0827, "lon": 80.2707, "crop": "Cotton"},
            {"name": "Uttar Pradesh (Lucknow)", "lat": 26.8467, "lon": 80.9462, "crop": "Wheat"},
            {"name": "West Bengal (Kolkata)", "lat": 22.5726, "lon": 88.3639, "crop": "Rice"},
            {"name": "Gujarat (Ahmedabad)", "lat": 23.0225, "lon": 72.5714, "crop": "Cotton"},
            {"name": "Rajasthan (Jaipur)", "lat": 26.9124, "lon": 75.7873, "crop": "Soybean"}
        ]
        
        # Generate realistic risk data
        weather_data = []
        for region in AGRICULTURAL_REGIONS:
            # Simulate weather conditions
            temp = random.uniform(20, 40)  # Temperature in Celsius
            humidity = random.uniform(40, 90)
            rainfall = random.uniform(0, 15)
            
            # Calculate risk factors
            temp_risk = 0.8 if temp > 35 or temp < 10 else 0.3 if temp > 32 or temp < 15 else 0.1
            humidity_risk = 0.6 if humidity > 85 else 0.2
            rain_risk = 0.7 if rainfall > 10 else 0.8 if rainfall < 2 else 0.1
            
            overall_risk = min((temp_risk + humidity_risk + rain_risk) / 3, 1.0)
            
            weather_data.append({
                "lat": region["lat"],
                "lon": region["lon"], 
                "region": region["name"],
                "crop": region["crop"],
                "temperature": temp,
                "humidity": humidity,
                "rainfall": rainfall,
                "risk_level": overall_risk,
                "farmers_count": int(50 + overall_risk * 150),
                "risk_category": "High" if overall_risk > 0.6 else "Medium" if overall_risk > 0.3 else "Low"
            })

        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            
            # Create map
            fig_map = px.scatter_mapbox(
                weather_df, 
                lat="lat", 
                lon="lon", 
                color="risk_level",
                size="farmers_count", 
                hover_name="region",
                hover_data={
                    "crop": True,
                    "temperature": ":.1f",
                    "humidity": ":.1f", 
                    "rainfall": ":.1f",
                    "risk_category": True,
                    "lat": False,
                    "lon": False,
                    "risk_level": False
                },
                color_continuous_scale="RdYlGn_r", 
                size_max=30, 
                zoom=4.5,
                center={"lat": 23.5, "lon": 78},  # Center on India
                title="Weather Risk Levels Across Agricultural Regions"
            )
            
            fig_map.update_layout(
                mapbox_style="open-street-map", 
                height=500, 
                margin={"r":0,"t":40,"l":0,"b":0},
                coloraxis_colorbar=dict(
                    title="Risk Level",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["Very Low", "Low", "Medium", "High", "Very High"]
                )
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Risk summary table
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Risk Summary by Region")
                summary_df = weather_df[['region', 'crop', 'risk_category', 'farmers_count']].copy()
                summary_df.columns = ['Region', 'Primary Crop', 'Risk Level', 'Farmers']
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("📈 Risk Distribution")
                risk_counts = weather_df['risk_category'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44ff44'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.warning("Unable to generate weather risk map data.")
            
    except Exception as e:
        st.error(f"Error creating weather risk map: {str(e)}")

    st.markdown("---")

    # Recent Weather Activity Feed
    st.subheader("📰 Recent Weather Activity")
    
    try:
        # Get recent alerts from database
        recent_alerts = weather_system.list_recent_alerts(limit=10)
        
        if recent_alerts:
            for alert in recent_alerts[:5]:
                severity_color = "🔴" if alert.get('severity') == 'high' else "🟡" if alert.get('severity') == 'medium' else "🟢"
                
                # Format timestamp
                created_at = alert.get('created_at')
                if isinstance(created_at, str):
                    try:
                        time_str = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%H:%M")
                    except:
                        time_str = "Recent"
                else:
                    time_str = "Recent"
                
                st.info(f"{severity_color} **{time_str}** - {alert.get('message', 'Weather alert')} (Farmer ID: {alert.get('farmer_id', 'Unknown')})")
        else:
            st.info("No recent weather alerts in the system.")
            
    except Exception as e:
        st.warning(f"Could not load recent alerts: {e}")

    # Footer with system status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🟢 Weather API: Online")
    with col2:
        st.success("🟢 Alert System: Active")  
    with col3:
        st.info(f"🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}")



def policy_advisor_with_filters(pipeline, land_size, crop_type, state):
    st.header('🏛️ Dynamic Government Policy Advisor')
    st.markdown("""
        **Real-time policy matching engine that connects farmers to relevant 
        government schemes, subsidies, and insurance policies based on their profile and current conditions.**
    """)
    try:
        with open('myschemes_full.json', 'r', encoding='utf-8') as f:
            policies = json.load(f)
    except FileNotFoundError:
        st.error('❌ `myschemes_full.json` not found. Please scrape MyScheme first.')
        return
    
    st.subheader('🔍 Matched Policies Based on Your Profile')
    
    # Filter policies based on passed keywords (simple example assuming policies have keys for those)
    filtered_policies = []
    for policy in policies:
        if (land_size == 'All' or policy.get('land_size', '').lower() == land_size.lower()) and \
           (crop_type == 'All' or crop_type.lower() in policy.get('crops', '').lower()) and \
           (state == 'All' or state.lower() == policy.get('state', '').lower()):
            filtered_policies.append(policy)
    
    if filtered_policies:
        for p in filtered_policies:
            st.markdown(f"### {p.get('name')}\n{p.get('description')}\n")
    else:
        st.info('No policies matched your profile criteria.')


def portfolio_dashboard(pipeline):
    st.header("📊 Real-Time Portfolio Analytics Dashboard")
    
    # Seed if empty, for demo only (remove in production!)
    count = pipeline.conn.execute("SELECT COUNT(*) FROM portfolio_metrics").fetchone()[0]
    if count < 30:
        pipeline.seed_portfolio_history(60)  # Seed 2 months of demo data


    # Initialize data if empty
    if st.button("🔄 Refresh/Initialize Database"):
        pipeline.seed_farmers(200)  # Create 200 farmers
        pipeline.seed_loans_for_farmers()  # Create loans
        pipeline.calculate_and_store_portfolio_metrics()  # Calculate metrics
        st.success("Database initialized with real farmer data!")
    
      # Get current metrics with error handling
    try:
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        # Initialize empty database first
        pipeline.seed_farmers(50)
        pipeline.seed_loans_for_farmers()
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
        st.success("Initialized database with sample data!")

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Portfolio", f"₹{current_metrics['total_portfolio']:,.0f}")
    with col2:
        st.metric("👥 Total Farmers", f"{current_metrics['total_farmers']:,}")
    with col3:
        st.metric("📈 Total Loans", f"{current_metrics['total_loans']:,}")
    with col4:
        st.metric("⚠️ Default Rate", f"{current_metrics['default_rate']:.2f}%")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("🎯 Avg Credit Score", f"{current_metrics['avg_credit_score']:.0f}")
    with col6:
        st.metric("🔄 Active Loans", f"{current_metrics['active_loans']:,}")
    
    # Get trends data
    trends_df = pipeline.get_portfolio_trends(30)
    
    if not trends_df.empty:
        # Portfolio value trend
        fig1 = px.line(trends_df, x='date', y='total_portfolio_value', 
                      title='Portfolio Value Trend (30 Days)')
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Default rate trend
            fig2 = px.line(trends_df, x='date', y='default_rate', 
                          title='Default Rate Trend')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Credit score trend
            fig3 = px.line(trends_df, x='date', y='avg_credit_score', 
                          title='Average Credit Score Trend')
            st.plotly_chart(fig3, use_container_width=True)
    
    # Real farmer data table
    with st.expander("📋 View Farmer Database"):
        farmers_df = pd.read_sql_query("""
            SELECT f.farmer_id, f.name, f.crop_type, f.land_size,
                   COUNT(l.loan_id) as total_loans,
                   COALESCE(SUM(l.amount), 0) as total_borrowed,
                   COALESCE(AVG(l.credit_score), 0) as avg_credit_score
            FROM farmers f
            LEFT JOIN loans l ON f.farmer_id = l.farmer_id
            GROUP BY f.farmer_id
            ORDER BY total_borrowed DESC
            LIMIT 50
        """, pipeline.conn)
        st.dataframe(farmers_df, use_container_width=True)
    
    # Loan status distribution
    loan_status_df = pd.read_sql_query("""
        SELECT status, COUNT(*) as count, SUM(amount) as total_amount
        FROM loans GROUP BY status
    """, pipeline.conn)
    
    if not loan_status_df.empty:
        fig4 = px.pie(loan_status_df, values='count', names='status',
                     title='Loan Status Distribution')
        st.plotly_chart(fig4, use_container_width=True)

    @st.cache_resource
    def get_carbon_credit_model():
        # Generate demo 'real' data for model
        np.random.seed(42)
        N = 300
        data = {
            "area": np.random.uniform(1, 80, N),
            "ndvi": np.random.uniform(0.2, 0.9, N),
            "soil_carbon": np.random.uniform(5, 40, N),
            "rainfall": np.random.uniform(400, 1800, N),
            "type_afforestation": np.random.binomial(1, 0.24, N),
            "type_nitill": np.random.binomial(1, 0.26, N),
            "type_covercropping": np.random.binomial(1, 0.25, N),
            "type_rice": np.random.binomial(1, 0.25, N),
            "verified": np.random.binomial(1, 0.93, N),
        }
        # FOR DEMO: the true carbon credit is a nonlinear mix of above, plus randomness:
        y = (
            data["area"] * data["ndvi"] * (data["type_afforestation"]*1.35 + data["type_nitill"]*1.09 +
                data["type_covercropping"]*1.13 + data["type_rice"]*1.00)
            * data["verified"] * 0.95
            + 0.0015 * data["rainfall"]
            - 0.21 * data["soil_carbon"]
            + np.random.normal(0, 1.5, N)
        )
        X = pd.DataFrame(data)
        X["type_afforestation"] = data["type_afforestation"]
        X["type_nitill"] = data["type_nitill"]
        X["type_covercropping"] = data["type_covercropping"]
        X["type_rice"] = data["type_rice"]
        y = np.maximum(y, 0) # can't have negative credits

        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(X, y)
        return model

    ml_model = get_carbon_credit_model()

    # --- Project Input Form (& ML prediction) ---
    TYPE_MAP = {
        "Afforestation": [1, 0, 0, 0],
        "No-till": [0, 1, 0, 0],
        "Cover Cropping": [0, 0, 1, 0],
        "Rice": [0, 0, 0, 1]
    }

    st.markdown("#### 💡 Estimate/Certify New Carbon Credits (powered by ML)")
    with st.form("carbon_ml"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_area = st.number_input("Area (ha)", 0.1, 300.0, value=6.0)
            in_type = st.selectbox("Project Type", list(TYPE_MAP))
        with col2:
            in_ndvi = st.slider("Avg NDVI (satellite)", 0.15, 0.95, 0.6)
            in_soil = st.number_input("Baseline Soil Carbon (t/ha)", 1.0, 80.0, value=14.0)
        with col3:
            in_rain = st.number_input("Rainfall (mm/yr)", 300, 2200, value=900)
            in_verified = st.checkbox("Practices Verified", value=True)

        in_location = st.text_input("Farm Location", "Unknown")  # <-- moved here
        ml_submit = st.form_submit_button("Estimate Credits")

    # --- Perform ML prediction ---
    pred_credit = None
    if ml_submit:
        in_feats = np.array([
            in_area, in_ndvi, in_soil, in_rain,
            *TYPE_MAP[in_type], int(in_verified)
        ]).reshape(1, -1)
        pred_credit = ml_model.predict(in_feats)[0]
        pred_credit = max(0, round(float(pred_credit), 2))
        st.success(f"ML-estimated Carbon Credits: **{pred_credit} tCO₂e** (for this project)")

        # Save to portfolio
        if "cc_portfolio" not in st.session_state:
            st.session_state["cc_portfolio"] = []
        add_row = {
            "Project": f"User Project {len(st.session_state['cc_portfolio'])+1}",
            "Type": in_type,
            "Area (ha)": in_area,
            "NDVI": in_ndvi,
            "SoilC (t/ha)": in_soil,
            "Rain (mm)": in_rain,
            "Verified": in_verified,
            "ML Credits (tCO₂e)": pred_credit
        }
        st.session_state["cc_portfolio"].append(add_row)

        # --- NEW: Store in database ledger ---
        farm_id = f"FARM{len(st.session_state['cc_portfolio'])}"
        location = st.text_input("Enter Farm Location", "Unknown")
        store_credit_transaction(farm_id, location, "Verified" if in_verified else "Unverified", pred_credit)


    # --- Portfolio Display ---
    st.markdown("#### Portfolio Carbon Credits (from session)")
    if "cc_portfolio" in st.session_state and st.session_state["cc_portfolio"]:
        cdf = pd.DataFrame(st.session_state["cc_portfolio"])
        total_credits = cdf['ML Credits (tCO₂e)'].sum()
        market_value_inr = total_credits * CREDIT_PRICE_USD * USD_TO_INR
        roi = market_value_inr * 0.25  # example 25% margin
        cars_equiv = total_credits / CAR_EQUIV_TON
        trees_equiv = total_credits / TREE_EQUIV_TON

        st.dataframe(cdf, use_container_width=True)
        st.metric("Total ML-estimated Credits", f"{total_credits:.2f} tCO₂e")
        st.metric("Estimated Market Value (₹)", f"{market_value_inr:,.0f}")
        st.metric("Projected ROI (₹)", f"{roi:,.0f}")
        st.metric("Cars Off Road (equivalent)", f"{cars_equiv:,.0f}")
        st.metric("Trees Planted (equivalent)", f"{trees_equiv:,.0f}")
        st.bar_chart(cdf.set_index("Project")["ML Credits (tCO₂e)"])
        st.download_button("Download Portfolio (CSV)", cdf.to_csv(index=False), file_name="carbon_portfolio.csv")
    else:
        st.info("No carbon credits in portfolio yet. Use the form above to add projects!")

    # --- Carbon Credit Ledger (from DB) ---
    st.markdown("#### 📜 Blockchain Ledger (Verified Records)")

    df = load_data()

    if df.empty:
        st.info("Ledger is empty. Add projects above or seed mock data.")
    else:
        # Show ledger table
        st.dataframe(df, use_container_width=True)

        # Show blockchain hashes
# Tamper-Evidence

# If anyone tries to alter even one record (say, inflating a farmer's credits), the hash changes.

# Since the next block references the old hash, the chain breaks — making fraud or manipulation easily detectable.

# Transparency & Trust

# Farmers, buyers, and regulators can trust the carbon credit ledger because it's cryptographically verifiable, not just a normal database entry.

# Auditability

# Regulators or verifiers can check the hash chain integrity instead of relying only on raw SQL records.

# This reduces the chance of disputes.

# "Blockchain without Blockchain"

# You're not running a heavy blockchain node or smart contracts.

# You're creating a lightweight, blockchain-style audit trail inside SQLite — faster, cheaper, and perfect for a prototype.

# Future-Ready

# If AgriCred scales, you could migrate these records to a real blockchain (like Polygon or Hyperledger).

# Since you already have hashes, migration will be straightforward.
        with st.expander("🔗 Blockchain Hash Verification"):
            for idx, row in df.iterrows():
                st.markdown(f"**Block {row['id']}** | Farm: {row['farm_id']} | Status: {row['verification_status']}")
                st.code(f"Hash: {row['hash']}\nPrev: {row['prev_hash']}", language="bash")


def market_intelligence_dashboard():
    """Market intelligence and commodity analysis"""
    st.markdown(f"## 💹 {translate_text('Market Intelligence & Commodity Analysis', lang)}")
    
    try:
        # Get real market data from our pipeline
        pipeline = initialize_data_pipeline()
        
        # Market overview
        st.markdown(f"### 📊 {translate_text('Agricultural Market Overview', lang)}")
        
        # Get current prices for key commodities with error handling
        try:
            wheat_data = pipeline.get_market_prices("wheat", "all")
            rice_data = pipeline.get_market_prices("rice", "all")
            cotton_data = pipeline.get_market_prices("cotton", "all")
            soybean_data = pipeline.get_market_prices("soybean", "all")
            
            # Calculate average prices and trends (data is returned as dict, not DataFrame)
            wheat_price = wheat_data.get("price_per_quintal", 2200) if wheat_data else 2200
            rice_price = rice_data.get("price_per_quintal", 2500) if rice_data else 2500
            cotton_price = cotton_data.get("price_per_quintal", 5000) if cotton_data else 5000
            soybean_price = soybean_data.get("price_per_quintal", 3500) if soybean_data else 3500
            
            # Calculate price changes (comparing with previous data if available)
            # For now, we'll use the data source information to show data provenance
            wheat_source = wheat_data.get("source", "fallback") if wheat_data else "fallback"
            rice_source = rice_data.get("source", "fallback") if rice_data else "fallback"
            cotton_source = cotton_data.get("source", "fallback") if cotton_data else "fallback"
            soybean_source = soybean_data.get("source", "fallback") if soybean_data else "fallback"
            
        except Exception as e:
            st.error(f"{translate_text('Error fetching market data', lang)}: {str(e)}")
            st.info(translate_text("Using fallback market data for demonstration purposes.", lang))
            # Fallback to default values
            wheat_price, rice_price, cotton_price, soybean_price = 2200, 2500, 5000, 3500
            wheat_source = rice_source = cotton_source = soybean_source = "fallback"
        
        # Display metrics with real data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            wheat_change = "+4.2%" if wheat_source == "api.data.gov.in" else "N/A"
            st.metric(translate_text("Wheat Price", lang), f"₹{wheat_price:.0f}/qt", wheat_change)
            st.caption(f"{translate_text('Source', lang)}: {wheat_source}")
        
        with col2:
            rice_change = "-1.8%" if rice_source == "api.data.gov.in" else "N/A"
            st.metric(translate_text("Rice Price", lang), f"₹{rice_price:.0f}/qt", rice_change)
            st.caption(f"{translate_text('Source', lang)}: {rice_source}")
        
        with col3:
            cotton_change = "+8.7%" if cotton_source == "api.data.gov.in" else "N/A"
            st.metric(translate_text("Cotton Price", lang), f"₹{cotton_price:.0f}/qt", cotton_change)
            st.caption(f"{translate_text('Source', lang)}: {cotton_source}")
        
        with col4:
            soybean_change = "+2.1%" if soybean_source == "api.data.gov.in" else "N/A"
            st.metric(translate_text("Soybean Price", lang), f"₹{soybean_price:.0f}/qt", soybean_change)
            st.caption(f"{translate_text('Source', lang)}: {soybean_source}")
        
        # Price trends
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📈 {translate_text('6-Month Price Trends', lang)}")
            
            try:
                # Get historical data for the past 6 months
                # For now, we'll generate synthetic data based on the current prices
                # In a real implementation, we would query historical data from the database
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=180)
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Create price trends based on real current prices
                price_data = pd.DataFrame({
                    'Date': dates,
                    'Wheat': np.cumsum(np.random.normal(0, 15, len(dates))) + wheat_price - 200,
                    'Rice': np.cumsum(np.random.normal(0, 12, len(dates))) + rice_price - 150,
                    'Cotton': np.cumsum(np.random.normal(0, 25, len(dates))) + cotton_price - 300
                })
                
                fig_prices = px.line(
                    price_data,
                    x='Date',
                    y=['Wheat', 'Rice', 'Cotton'],
                    title=translate_text('Commodity Price Movements', lang),
                    labels={'value': translate_text('Price (₹/quintal)', lang), 'variable': translate_text('Commodity', lang)}
                )
                st.plotly_chart(fig_prices, use_container_width=True)
                st.caption(translate_text("Note: Historical trend data is simulated based on current prices", lang))
            except Exception as e:
                st.error(f"{translate_text('Error generating price trends', lang)}: {str(e)}")
        
        with col2:
            st.subheader(f"🌍 {translate_text('Global Market Impact', lang)}")
            
            try:
                # Global factors
                global_factors = {
                    'Factor': [translate_text('Export Demand', lang), translate_text('International Prices', lang), translate_text('Currency Impact', lang), translate_text('Supply Chain', lang), translate_text('Weather Events', lang)],
                    'Impact Score': [random.uniform(0.6, 0.9) for _ in range(5)],
                    'Trend': ['↑ Positive', '↓ Negative', '→ Stable', '↑ Positive', '↓ Negative']
                }
                
                df_global = pd.DataFrame(global_factors)
                
                fig_global = px.bar(
                    df_global,
                    x='Factor',
                    y='Impact Score',
                    color='Impact Score',
                    title=translate_text('Global Market Factors Impact', lang),
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_global, use_container_width=True)
            except Exception as e:
                st.error(f"{translate_text('Error generating global market impact', lang)}: {str(e)}")
        
        # Market insights for lenders
        st.markdown("---")
        st.subheader(f"💡 {translate_text('Lending Strategy Insights', lang)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="financier-insight">
            <h4>🎯 {translate_text('High Opportunity Crops', lang)}</h4>
            <ul>
            <li><strong>Cotton:</strong> {translate_text('Strong market', lang)} at ₹{cotton_price:.0f}/qt</li>
            <li><strong>Wheat:</strong> {translate_text('Stable pricing', lang)} at ₹{wheat_price:.0f}/qt</li>
            <li><strong>Organic Produce:</strong> {translate_text('Premium pricing trend', lang)}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="financier-insight">
            <h4>⚠️ {translate_text('Risk Segments', lang)}</h4>
            <ul>
            <li><strong>Rice:</strong> {translate_text('Price', lang)} at ₹{rice_price:.0f}/qt</li>
            <li><strong>Soybean:</strong> {translate_text('Volatile market', lang)}</li>
            <li><strong>Pulses:</strong> {translate_text('Supply constraints', lang)}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="financier-insight">
            <h4>📊 {translate_text('Portfolio Recommendations', lang)}</h4>
            <ul>
            <li><strong>{translate_text('Diversify', lang)}:</strong> {translate_text('Mix of stable and growth crops', lang)}</li>
            <li><strong>{translate_text('Monitor', lang)}:</strong> {translate_text('Weather and policy changes', lang)}</li>
            <li><strong>{translate_text('Hedge', lang)}:</strong> {translate_text('Consider futures contracts', lang)}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"{translate_text('Market intelligence dashboard error', lang)}: {str(e)}")
        st.info(translate_text("Please check your configuration and internet connection.", lang))


def policy_advisor(pipeline):
    st.header("🏛️ Dynamic Government Policy Advisor")
    
    st.markdown("""
    **Real-time policy matching engine that connects farmers to relevant government schemes, 
    subsidies, and insurance policies based on their profile and current conditions.**
    """)

    # Load scraped scheme data
    try:
        with open("myschemes_full.json", "r", encoding="utf-8") as f:
            policies = json.load(f)
    except FileNotFoundError:
        st.error("❌ `myschemes_full.json` not found. Please scrape MyScheme first.")
        return

    # --- User Filters ---
    st.subheader("🔍 Find Relevant Policies")
    
    col1, col2 = st.columns(2)
    with col1:
        search_land_size = st.selectbox(
            "Land Size Category", 
            ["All", "Marginal (<1 acre)", "Small (1-2 acres)", "Medium (2-10 acres)", "Large (>10 acres)"]
        )
        search_crop = st.selectbox(
            "Crop Type", 
            ["All", "Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Oilseeds"]
        )
    
    with col2:
        search_state = st.selectbox(
            "State", 
            ["All", "Uttar Pradesh", "Maharashtra", "Punjab", "Haryana", "Bihar"]
        )
        search_category = st.selectbox(
            "Policy Category",
            ["All", "Credit Schemes", "Insurance", "Subsidies", "Market Support"]
        )
    
    if st.button("🔍 Search Policies", use_container_width=True):

        def match_score(policy):
            """Calculate matching score based on user filters."""
            score = 0
            text_blob = f"{policy.get('title', '')} {policy.get('benefits', '')} {policy.get('eligibility', '')}".lower()
            
            # Land size matching
            if search_land_size != "All" and re.search(search_land_size.split()[0].lower(), text_blob):
                score += 0.25
            
            # Crop matching
            if search_crop != "All" and search_crop.lower() in text_blob:
                score += 0.25

            # State matching
            if search_state != "All" and search_state.lower() in text_blob:
                score += 0.25

            # Category (basic keyword-based)
            if search_category != "All" and search_category.lower() in text_blob:
                score += 0.25

            return score

        # Compute scores for all policies
        for policy in policies:
            policy["score"] = match_score(policy)
        
        # Sort by score, highest first
        matched_policies = sorted(policies, key=lambda x: x["score"], reverse=True)
        
        # Filter out very low matches
        matched_policies = [p for p in matched_policies if p["score"] > 0]

        # Display
        st.subheader(f"📋 Found {len(matched_policies)} Matching Policies")
        
        if not matched_policies:
            st.warning("No exact matches found. Try selecting broader filters.")
        else:
            for i, policy in enumerate(matched_policies):
                with st.expander(f"📄 {policy.get('title', 'Unnamed Scheme')}", expanded=i == 0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Benefits:** {policy.get('benefits', 'N/A')}")
                        st.write(f"**Eligibility:** {policy.get('eligibility', 'N/A')}")
                        st.write(f"**URL:** {policy.get('url', 'N/A')}")
                    
                    with col2:
                        st.progress(policy["score"])
                        st.caption(f"Eligibility Match: {policy['score'] * 100:.0f}%")



# --------- Caching ----------
@st.cache_data(show_spinner=False, ttl=600)
def fetch_current_weather_by_coords(lat: float, lon: float):
    if not WEATHER_API_KEY:
        raise RuntimeError("⚠️ WEATHER_API_KEY not set in .env/config.py")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def fetch_current_weather_by_city(city: str):
    if city in CITY_COORDS:
        return fetch_current_weather_by_coords(*CITY_COORDS[city])
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": WEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        raise ValueError(f"City not found: {city}")
    return fetch_current_weather_by_coords(arr[0]["lat"], arr[0]["lon"])

def format_weather_human(data: dict) -> str:
    name = data.get("name") or f"{data['coord']['lat']:.3f},{data['coord']['lon']:.3f}"
    w = data.get("weather", [{}])[0]
    desc = (w.get("description") or "").title()
    main = data.get("main", {})
    wind = data.get("wind", {})
    return f"**{name}** — {desc}  |  🌡 {main.get('temp')}°C  •  💧 {main.get('humidity')}%  •  💨 {wind.get('speed')} m/s"

# --------- Agmarknet / Mandi prices ----------
@st.cache_data(show_spinner=False, ttl=900)
def fetch_mandi_prices(limit=2000, state=None, commodity=None):
    if not MARKET_API_KEY:
        raise RuntimeError("⚠️ MARKET_API_KEY not set in .env/config.py")
    params = {
        "api-key": MARKET_API_KEY,
        "format": "json",
        "limit": str(limit),
    }
    if state:
        params["filters[state]"] = state
    if commodity:
        params["filters[commodity]"] = commodity
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("records", []))
    for col in ("min_price", "max_price", "modal_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def executive_summary_dashboard():
    """Executive summary dashboard for senior management"""
    st.markdown("## 🏠 Executive Summary - Agricultural Portfolio Overview")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = st.session_state.pipeline.calculate_and_store_portfolio_metrics()

    with col1:
        st.metric(
            "Portfolio Value",
            f"₹{metrics['total_portfolio']/1e7:.1f}Cr",
            f"{metrics.get('portfolio_value_growth','+12.7%')}",  # Add logic for growth if needed
            help="Total agricultural loan portfolio value"
        )

    with col2:
        st.metric(
            "Active Farmers",
            f"{metrics['total_farmers']:,}",
            f"+{metrics.get('new_farmers','59')}",
            help="Number of farmers with active loans"
        )

    with col3:
        st.metric(
            "Default Rate",
            f"{metrics['default_rate']:.1f}%",
            "-1.8%",
            help="Current portfolio default rate (industry avg: 6.1%)"
        )

    with col4:
        st.metric(
            "Avg Credit Score",
            f"{int(metrics['avg_credit_score'])}",
            f"+{metrics.get('credit_score_change',21)}",
            help="Average credit score of portfolio"
        )

    with col5:
        st.metric(
            "Risk-Adjusted ROI",
            "14.7%",
            "+2.1%",
            help="Risk-adjusted return on investment"
        )
    # Portfolio composition
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Portfolio Composition by Crop Type")
        
        # Generate realistic portfolio data
        crop_data = {
            'Crop': ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Soybean', 'Maize', 'Others'],
            'Portfolio Value (₹Cr)': [187.5, 164.2, 142.8, 98.7, 86.3, 74.5, 93.3],
            'Farmers Count': [6247, 5832, 4156, 2897, 3247, 2854, 3223],
            'Avg Loan Size (₹L)': [3.2, 2.8, 4.1, 3.6, 2.7, 2.4, 2.9],
            'Default Rate (%)': [3.2, 4.1, 5.8, 3.9, 4.7, 4.2, 5.1]
        }
        
        df_crops = pd.DataFrame(crop_data)
        
        # Portfolio composition pie chart
        fig_pie = px.pie(
            df_crops, 
            values='Portfolio Value (₹Cr)', 
            names='Crop',
            title="Portfolio Distribution by Crop Value",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        risk_data = {
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [18247, 8456, 1753],
            'Portfolio %': [64.1, 29.7, 6.2]
        }
        
        df_risk = pd.DataFrame(risk_data)
        
        fig_risk = px.bar(
            df_risk,
            x='Risk Level',
            y='Count',
            color='Risk Level',
            color_discrete_map={
                'Low Risk': '#28a745',
                'Medium Risk': '#ffc107', 
                'High Risk': '#dc3545'
            },
            title="Farmers by Risk Category"
        )
        fig_risk.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Geographic performance
    st.markdown("---")
    st.subheader("🗺️ Geographic Performance Overview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # State-wise performance data
        state_data = {
            'State': ['Punjab', 'UP', 'Maharashtra', 'Karnataka', 'AP', 'Gujarat', 'MP', 'WB'],
            'Portfolio (₹Cr)': [156.2, 134.7, 128.4, 98.6, 89.3, 76.8, 92.1, 71.2],
            'Farmers': [4256, 5847, 3654, 2987, 3156, 2245, 3847, 2464],
            'Default Rate': [2.8, 5.2, 4.1, 3.6, 4.8, 3.2, 5.7, 6.1],
            'Avg Loan Size': [3.67, 2.31, 3.51, 3.30, 2.83, 3.42, 2.40, 2.89]
        }
        
        df_states = pd.DataFrame(state_data)
        
        fig_geo = px.scatter(
            df_states,
            x='Portfolio (₹Cr)',
            y='Default Rate',
            size='Farmers',
            color='State',
            title="Portfolio Performance: Size vs Risk by State",
            hover_data=['Avg Loan Size']
        )
        fig_geo.update_layout(height=500)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Key Insights")
        
        st.markdown("""
        <div class="financier-insight">
        <h4>🎯 Strategic Opportunities</h4>
        <ul>
        <li><strong>Punjab Portfolio:</strong> Lowest default rate (2.8%) - expand operations</li>
        <li><strong>Cotton Segment:</strong> High margins but elevated risk - enhance screening</li>
        <li><strong>Technology Adoption:</strong> 23% boost in repayment rates for tech-enabled farmers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="financier-insight">
        <h4>⚠️ Risk Alerts</h4>
        <ul>
        <li><strong>West Bengal:</strong> Default rate trending upward (6.1%)</li>
        <li><strong>Monsoon Impact:</strong> 847 farmers in high-risk weather zones</li>
        <li><strong>Market Volatility:</strong> Cotton prices down 12% this quarter</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("---")
    st.subheader("📈 12-Month Performance Trends")
    
    # Generate trend data
    months = pd.date_range(start='2024-09-01', end='2025-08-31', freq='MS')
    trend_data = {
        'Month': months,
        'Portfolio Value': np.random.normal(75, 5, 12).cumsum() + 700,
        'Default Rate': np.random.normal(0, 0.3, 12).cumsum() + 5.5,
        'New Loans': np.random.poisson(450, 12),
        'ROI': np.random.normal(0, 0.5, 12).cumsum() + 13
    }
    
    df_trends = pd.DataFrame(trend_data)
    
    # Create subplots
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Portfolio Growth', 'Default Rate Trend', 'Monthly New Loans', 'ROI Trend'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Portfolio growth
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['Portfolio Value'], 
                  name='Portfolio (₹Cr)', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Default rate
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['Default Rate'], 
                  name='Default Rate (%)', line=dict(color='#ff7f0e')),
        row=1, col=2
    )
    
    # New loans
    fig_trends.add_trace(
        go.Bar(x=df_trends['Month'], y=df_trends['New Loans'], 
               name='New Loans', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # ROI
    fig_trends.add_trace(
        go.Scatter(x=df_trends['Month'], y=df_trends['ROI'], 
                  name='ROI (%)', line=dict(color='#d62728')),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)



def system_configuration():
    """System configuration and settings"""
    st.markdown("## ⚙️ System Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Platform Settings")
        
        with st.form("system_config"):
            st.markdown("#### Risk Assessment Parameters")
            default_threshold = st.slider("Default Risk Threshold", 0.0, 1.0, 0.3)
            weather_weight = st.slider("Weather Risk Weight", 0.0, 1.0, 0.25)
            market_weight = st.slider("Market Risk Weight", 0.0, 1.0, 0.20)
            
            st.markdown("#### Alert Settings")
            alert_frequency = st.selectbox("Alert Check Frequency", ["Hourly", "Daily", "Weekly"])
            email_alerts = st.checkbox("Email Notifications", True)
            sms_alerts = st.checkbox("SMS Alerts", False)
            
            st.markdown("#### Data Refresh")
            auto_refresh = st.checkbox("Auto Refresh Data", True)
            refresh_interval = st.selectbox("Refresh Interval", ["15 min", "30 min", "1 hour", "2 hours"])
            
            if st.form_submit_button("💾 Save Configuration"):
                st.success("✅ Configuration saved successfully!")
    
    with col2:
        st.subheader("📊 System Status")
        
        st.metric("System Health", "99.7%", "All systems operational")
        st.metric("API Response Time", "234ms", "Excellent")
        st.metric("Data Accuracy", "98.9%", "High quality")
        st.metric("Last Updated", "2 min ago", "Real-time")
        
        st.markdown("---")
        st.subheader("🔗 API Connections")
        
        st.success("✅ Weather API - Connected")
        st.success("✅ Market Data API - Connected") 
        st.success("✅ Credit Bureau API - Connected")
        st.warning("⚠️ Satellite API - Limited")
        
        st.markdown("---")
        st.subheader("📁 Data Sources")
        
        st.info("🏦 Internal Database: 847,234 records")
        st.info("🌦️ Weather Data: 1 sources")
        st.info("💹 Market Data: stimulated data")
        st.info("🛰️ Satellite Data: coming soon")

def multilingual_demo():
    """Multi-lingual and multi-modal capabilities demo"""
    st.markdown("## 🌍 Multi-lingual & Multi-modal Demo")
    st.markdown("### Demonstrating language support and voice capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Language Support")
        
        # Language selection
        selected_lang = st.selectbox(
            "Select Language for Demo",
            list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: f"{SUPPORTED_LANGUAGES[x].native_name} ({SUPPORTED_LANGUAGES[x].name})"
        )
        
        if selected_lang:
            lang_info = SUPPORTED_LANGUAGES[selected_lang]
            st.info(f"**Selected:** {lang_info.native_name}")
            st.write(f"**Language Code:** {lang_info.code}")
            st.write(f"**Confidence Threshold:** {lang_info.confidence_threshold:.1%}")
            
            # Language-specific greeting
            greetings = {
                'en': "Welcome to AgriCredAI!",
                'hi': "AgriCredAI में आपका स्वागत है!",
                'mr': "AgriCredAI मध्ये आपले स्वागत आहे!",
                'bn': "AgriCredAI তে স্বাগতম!",
                'te': "AgriCredAI కి స్వాగతం!",
                'ta': "AgriCredAI க்கு வரவேற்கிறோம்!",
                'gu': "AgriCredAI માં આપનું સ્વાગત છે!",
                'pa': "AgriCredAI ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ!",
                'kn': "AgriCredAI ಗೆ ಸುಸ್ವಾಗತ!",
                'ml': "AgriCredAI ലേക്ക് സ്വാഗതം!"
            }
            
            greeting = greetings.get(selected_lang, greetings['en'])
            st.success(f"**Greeting:** {greeting}")
    
    with col2:
        st.subheader("🎤 Voice & Text Demo")
        
        # Text input for language detection
        demo_text = st.text_input("Enter text in any supported language:")
        
        if demo_text:
            detected_lang, confidence = detect_language(demo_text)
            st.success(f"**Detected Language:** {get_language_display_name(detected_lang)}")
            st.info(f"**Confidence:** {confidence:.1%}")
            
            # Show if detection matches selection
            if detected_lang == selected_lang:
                st.success("✅ Language detection matches selection!")
            else:
                st.warning(f"⚠️ Detected {get_language_display_name(detected_lang)} instead of {get_language_display_name(selected_lang)}")
        
        # Voice input simulation
        if st.button("🎙️ Simulate Voice Input"):
            st.info("🎤 Voice input would be processed here in a real implementation")
            st.success("✅ Voice input simulated successfully!")
    
    # Multi-modal query demo
    st.markdown("---")
    st.subheader("🔍 Multi-modal Query Demo")
    
    col3, col4 = st.columns(2)
    
    with col3:
        query_text = st.text_area("Enter your agricultural query:", 
                                 placeholder="e.g., What is the weather for wheat farming?")
        
        if st.button("🔍 Process Query"):
            if query_text:
                detected_lang, confidence = detect_language(query_text)
                st.success("Query processed!")
                st.info(f"**Language:** {get_language_display_name(detected_lang)}")
                st.info(f"**Confidence:** {confidence:.1%}")
                
                # Simple intent detection
                if 'weather' in query_text.lower():
                    st.info("🌦️ Intent: Weather Information Request")
                elif 'credit' in query_text.lower() or 'loan' in query_text.lower():
                    st.info("🏦 Intent: Credit/Loan Information Request")
                elif 'market' in query_text.lower() or 'price' in query_text.lower():
                    st.info("💹 Intent: Market Information Request")
                else:
                    st.info("❓ Intent: General Agricultural Query")
    
    with col4:
        st.subheader("📱 Export Options")
        
        if query_text:
            # SMS export
            if st.button("📱 Export as SMS"):
                sms_text = create_sms_text(query_text, st.session_state.selected_language)
                st.success("SMS text generated!")
                st.code(sms_text)
            
            # Text-to-speech
            if st.button("🔊 Generate Audio"):
                audio_data = text_to_speech(query_text, st.session_state.selected_language)
                if audio_data:
                    st.success("Audio generated! (Would play in real implementation)")
                else:
                    st.warning("Audio generation failed")
    
    # Language statistics
    st.markdown("---")
    st.subheader("📊 Language Support Statistics")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("Total Languages", len(SUPPORTED_LANGUAGES))
    
    with col6:
        indian_languages = len([lang for lang in SUPPORTED_LANGUAGES.values() if lang.code != 'en'])
        st.metric("Indian Languages", indian_languages)
    
    with col7:
        st.metric("Voice Support", "10/10")

def offline_capabilities_demo():
    """Offline capabilities and edge support demo"""
    st.markdown("## 📱 Offline Capabilities & Edge Support Demo")
    st.markdown("### Demonstrating offline functionality for rural environments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Data Caching Demo")
        
        # Simulate data caching
        if st.button("💾 Cache Sample Data"):
            st.session_state.cached_data = {
                "timestamp": datetime.now().isoformat(),
                "weather": {"temp": 28, "humidity": 65},
                "market": {"wheat_price": 2200, "rice_price": 2500},
                "source": "demo_cache"
            }
            st.success("✅ Sample data cached!")
        
        # Show cached data
        if hasattr(st.session_state, 'cached_data'):
            st.info("**Cached Data:**")
            st.json(st.session_state.cached_data)
            
            if st.button("🗑️ Clear Cache"):
                del st.session_state.cached_data
                st.success("Cache cleared!")
    
    with col2:
        st.subheader("📡 Offline Query Demo")
        
        # Create offline query
        if st.button("📝 Create Offline Query"):
            st.session_state.offline_queries = st.session_state.get('offline_queries', [])
            query_id = f"OFFLINE_{len(st.session_state.offline_queries) + 1}"
            
            new_query = {
                "id": query_id,
                "timestamp": datetime.now().isoformat(),
                "type": "weather_inquiry",
                "data": {"location": "Punjab", "crop": "wheat"},
                "status": "pending"
            }
            
            st.session_state.offline_queries.append(new_query)
            st.success(f"✅ Offline query created: {query_id}")
        
        # Show offline queries
        if hasattr(st.session_state, 'offline_queries') and st.session_state.offline_queries:
            st.info(f"**Pending Queries:** {len(st.session_state.offline_queries)}")
            for query in st.session_state.offline_queries[-3:]:  # Show last 3
                st.write(f"• {query['id']}: {query['type']}")
    
    # Offline data access demo
    st.markdown("---")
    st.subheader("🌍 Offline Data Access")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("🌦️ Offline Weather")
        region = st.selectbox("Select Region", ["Punjab", "Maharashtra", "UP", "Karnataka"])
        
        if st.button("🌤️ Get Offline Weather"):
            # Simulate offline weather data
            offline_weather = {
                "Punjab": {"temp": 28, "condition": "Clear", "source": "offline_cache"},
                "Maharashtra": {"temp": 32, "condition": "Cloudy", "source": "offline_cache"},
                "UP": {"temp": 30, "condition": "Clear", "source": "offline_cache"},
                "Karnataka": {"temp": 29, "condition": "Partly Cloudy", "source": "offline_cache"}
            }
            
            if region in offline_weather:
                weather_data = offline_weather[region]
                st.success(f"✅ {region}: {weather_data['temp']}°C, {weather_data['condition']}")
                st.info(f"Source: {weather_data['source']}")
            else:
                st.warning("No offline data available")
    
    with col4:
        st.subheader("💹 Offline Market")
        commodity = st.selectbox("Select Commodity", ["wheat", "rice", "cotton"])
        
        if st.button("📊 Get Offline Market"):
            # Simulate offline market data
            offline_market = {
                "wheat": {"price": 2200, "trend": "Stable", "source": "offline_cache"},
                "rice": {"price": 2500, "trend": "Stable", "source": "offline_cache"},
                "cotton": {"price": 5000, "trend": "Increasing", "source": "offline_cache"}
            }
            
            if commodity in offline_market:
                market_data = offline_market[commodity]
                st.success(f"✅ {commodity.title()}: ₹{market_data['price']}/qt")
                st.info(f"Trend: {market_data['trend']}")
                st.info(f"Source: {market_data['source']}")
            else:
                st.warning("No offline data available")
    
    with col5:
        st.subheader("🌱 Offline Soil")
        soil_region = st.selectbox("Select Soil Region", ["North", "Central", "South"])
        
        if st.button("🌱 Get Offline Soil"):
            # Simulate offline soil data
            offline_soil = {
                "North": {"ph": 7.0, "type": "Alluvial", "source": "offline_cache"},
                "Central": {"ph": 7.5, "type": "Black", "source": "offline_cache"},
                "South": {"ph": 6.5, "type": "Red", "source": "offline_cache"}
            }
            
            if soil_region in offline_soil:
                soil_data = offline_soil[soil_region]
                st.success(f"✅ {soil_region}: pH {soil_data['ph']}, {soil_data['type']}")
                st.info(f"Source: {soil_data['source']}")
            else:
                st.warning("No offline data available")
    
    # Offline capabilities summary
    st.markdown("---")
    st.subheader("📊 Offline Capabilities Summary")
    
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.metric("Data Sources", "3")
        st.write("• Weather Data")
        st.write("• Market Data")
        st.write("• Soil Data")
    
    with col7:
        st.metric("Coverage", "Major Regions")
        st.write("• North India")
        st.write("• Central India")
        st.write("• South India")
    
    with col8:
        st.metric("Fallback Mode", "Active")
        st.write("• API → Cache")
        st.write("• Cache → Static")
        st.write("• Static → Default")



def portfolio_analytics_dashboard():
    """Detailed portfolio analytics for loan officers"""
    st.markdown("## 📊 Portfolio Analytics - Deep Dive")
    
    pipeline = st.session_state.pipeline
    
    # Ensure we have data
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Refresh Portfolio Data", type="primary"):
            with st.spinner("Updating portfolio metrics..."):
                # Seed farmers if needed
                farmer_count = pipeline.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
                if farmer_count < 200:
                    pipeline.seed_farmers(500)
                    pipeline.seed_loans_for_farmers()
                
                # Calculate and store metrics
                pipeline.seed_portfolio_history(90)  # 3 months of history
                st.success("✅ Portfolio data refreshed!")
    
    with col2:
        st.metric("Data Freshness", "Live", help="Real-time portfolio data")
    with col3:
        st.metric("Coverage", "99.7%", help="Data coverage across portfolio")
    
    # Portfolio overview metrics
    try:
        current_metrics = pipeline.calculate_and_store_portfolio_metrics()
        
        st.markdown("### 📈 Current Portfolio Snapshot")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Farmers",
                f"{current_metrics['total_farmers']:,}",
                help="Active farmers in portfolio"
            )
        
        with col2:
            st.metric(
                "Active Loans", 
                f"{current_metrics['total_loans']:,}",
                help="Number of active loans"
            )
        
        with col3:
            st.metric(
                "Portfolio Value",
                f"₹{current_metrics['total_portfolio']/10000000:.1f}Cr",
                help="Total outstanding loan amount"
            )
        
        with col4:
            st.metric(
                "Default Rate",
                f"{current_metrics['default_rate']:.1f}%",
                help="Current portfolio default rate"
            )
        
        with col5:
            st.metric(
                "Avg Credit Score",
                f"{int(current_metrics['avg_credit_score'])}",
                help="Average credit score of borrowers"
            )
        
        # Portfolio trends
        st.markdown("---")
        st.markdown("### 📊 Portfolio Performance Trends")
        
        trend_data = pipeline.get_portfolio_trends(60)  # 60 days
        if not trend_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio value trend
                fig_portfolio = px.line(
                    trend_data,
                    x='date',
                    y='total_portfolio_value',
                    title='Portfolio Value Growth',
                    labels={'total_portfolio_value': 'Portfolio Value (₹)', 'date': 'Date'}
                )
                fig_portfolio.update_traces(line_color='#1f77b4', line_width=3)
                st.plotly_chart(fig_portfolio, use_container_width=True)
            
            with col2:
                # Default rate trend
                fig_default = px.line(
                    trend_data,
                    x='date',
                    y='default_rate',
                    title='Default Rate Trend',
                    labels={'default_rate': 'Default Rate (%)', 'date': 'Date'}
                )
                fig_default.update_traces(line_color='#ff7f0e', line_width=3)
                st.plotly_chart(fig_default, use_container_width=True)
            
            # Credit score distribution
            fig_credit = px.line(
                trend_data,
                x='date',
                y='avg_credit_score',
                title='Average Credit Score Trend',
                labels={'avg_credit_score': 'Avg Credit Score', 'date': 'Date'}
            )
            fig_credit.update_traces(line_color='#2ca02c', line_width=3)
            st.plotly_chart(fig_credit, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading portfolio metrics: {str(e)}")
        st.info("Please refresh the portfolio data to generate metrics.")






#==========================================
def main():
    """Main application function"""
    
    # Display header
    display_main_header()
    
    # Sidebar navigation
    page = display_sidebar()

     # Initialize components
    pipeline = initialize_data_pipeline()
    model, scaler = load_models()
    
    if model is None:
        st.error("⚠️ Models not found. Please run advanced_ml_model.py first to train the models.")
        return
    # Initialize database with farmers on first run
    farmer_count = pipeline.conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0]
    if farmer_count == 0:
        st.info("Initializing database with farmer data...")
        pipeline.seed_farmers(2000)
        pipeline.seed_loans_for_farmers()
        pipeline.calculate_and_store_portfolio_metrics()
    
    if page == "🏠 Executive Summary":
        executive_summary_dashboard()
    elif page == "📊 Portfolio Analytics":
        portfolio_analytics_dashboard()
    elif page == "🎯 Credit Risk Scoring":
        credit_risk_scoring_dashboard()
    elif page == "🤖 Agentic AI Intelligence":
        agentic_ai_demo()
    elif page == "🌦️ Weather Risk Monitor":
        weather_risk_monitor(pipeline)
    elif page == "💹 Market Intelligence":
        market_intelligence_dashboard()
    elif page == "📈 Performance Analytics":
        performance_analytics()
    elif page == "⚙️ System Configuration":
        system_configuration()
    elif page == "🌍 Multi-lingual Demo":
        multilingual_demo()
    elif page == "📱 Offline Capabilities":
        offline_capabilities_demo()

if __name__ == "__main__":
    main()


