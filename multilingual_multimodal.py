"""
Multi-lingual and Multi-modal Support Module for AgriCredAI
Handles text/voice input in multiple Indian languages and provides TTS output
Implements speech-to-text, language detection, and text-to-speech capabilities
"""

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import numpy as np
import pandas as pd
import json
import logging
import os
import tempfile
import wave
import io
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from pathlib import Path
import requests
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageSupport:
    """Represents language support configuration"""
    code: str
    name: str
    native_name: str
    tts_voice: Optional[str] = None
    stt_model: Optional[str] = None
    confidence_threshold: float = 0.7
    fallback_language: str = "en"

@dataclass
class VoiceInput:
    """Represents voice input data"""
    audio_data: bytes
    language_detected: str
    confidence: float
    duration_seconds: float
    timestamp: str
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None

@dataclass
class MultiModalQuery:
    """Represents a multi-modal query"""
    query_id: str
    input_type: str  # "text", "voice", "mixed"
    detected_language: str
    confidence: float
    timestamp: str
    text_content: Optional[str] = None
    voice_content: Optional[VoiceInput] = None
    intent: Optional[str] = None
    entities: Optional[List[Dict]] = None

from typing import List


# Add this to multilingual_multimodal.py

try:
    from googletrans import Translator
    _translator = Translator()
except ImportError:
    _translator = None
import logging

logger = logging.getLogger(__name__)

_UI_TRANSLATIONS = {
    'en': {},
    'hi': {
        # Main Platform Headers
        'Capital One AgriCred AI Platform': 'कैपिटल वन एग्रीक्रेड एआई प्लेटफॉर्म',
        'Advanced Agricultural Credit Intelligence & Risk Management': 'उन्नत कृषि ऋण बुद्धिमत्ता और जोखिम प्रबंधन',
        'Empowering financial institutions with AI-driven insights for agricultural lending': 'कृषि ऋण के लिए एआई-संचालित अंतर्दृष्टि के साथ वित्तीय संस्थानों को सशक्त बनाना',
        
        # Navigation & Menu Items
        'Executive Summary': 'कार्यकारी सारांश',
        'Portfolio Analytics': 'पोर्टफोलियो विश्लेषण',
        'Credit Risk Scoring': 'क्रेडिट जोखिम स्कोरिंग',
        'Agentic AI Intelligence': 'एजेंटिक एआई बुद्धिमत्ता',
        'Weather Risk Monitor': 'मौसम जोखिम मॉनिटर',
        'Market Intelligence': 'बाजार बुद्धिमत्ता',
        'Performance Analytics': 'प्रदर्शन विश्लेषण',
        'System Configuration': 'सिस्टम कॉन्फ़िगरेशन',
        'Multi-lingual Demo': 'बहुभाषी डेमो',
        'Offline Capabilities': 'ऑफ़लाइन क्षमताएं',
        'Dashboard': 'डैशबोर्ड',
        'Home': 'होम',
        'Settings': 'सेटिंग्स',
        'Help': 'सहायता',
        'Profile': 'प्रोफ़ाइल',
        'Logout': 'लॉगआउट',
        
        # Language & Accessibility
        'Select Language': 'भाषा चुनें',
        'Current Language': 'वर्तमान भाषा',
        'Language': 'भाषा',
        'Accessibility': 'पहुंच',
        'Font Size': 'फ़ॉन्ट आकार',
        'High Contrast Mode': 'उच्च कंट्रास्ट मोड',
        
        # Dashboard Metrics & KPIs
        'Active Farmers': 'सक्रिय किसान',
        'Active Loans': 'सक्रिय ऋण',
        'High Risk Alerts': 'उच्च जोखिम चेतावनियाँ',
        'Rainfall Alerts': 'वर्षा चेतावनियाँ',
        'Safe Conditions': 'सुरक्षित परिस्थितियाँ',
        'Portfolio Value': 'पोर्टफोलियो मूल्य',
        'Default Rate': 'डिफ़ॉल्ट दर',
        'Avg Credit Score': 'औसत क्रेडिट स्कोर',
        'Total Revenue': 'कुल आय',
        'Net Profit': 'शुद्ध लाभ',
        'Cash Flow': 'नकदी प्रवाह',
        'Risk Assessment': 'जोखिम आकलन',
        'Loan Approval Rate': 'ऋण अनुमोदन दर',
        'NPAs': 'एनपीए',
        'Interest Rate': 'ब्याज दर',
        'Collateral Value': 'संपार्श्विक मूल्य',
        
        # Weather & Agricultural Terms
        'Live Weather Risk Monitoring System': 'लाइव मौसम जोखिम निगरानी प्रणाली',
        'Regional Weather Risk Map': 'क्षेत्रीय मौसम जोखिम मानचित्र',
        'Recent Weather Activity': 'हाल की मौसम गतिविधि',
        'Temperature': 'तापमान',
        'Humidity': 'आर्द्रता',
        'Wind Speed': 'हवा की गति',
        'Rainfall': 'वर्षा',
        'Soil Moisture': 'मिट्टी की नमी',
        'Crop Health': 'फसल स्वास्थ्य',
        'Irrigation': 'सिंचाई',
        'Harvest': 'फसल कटाई',
        'Sowing': 'बुआई',
        'Fertilizer': 'उर्वरक',
        'Pesticide': 'कीटनाशक',
        'Yield': 'उत्पादन',
        'Land Area': 'भूमि क्षेत्र',
        'Crop Type': 'फसल प्रकार',
        'Season': 'मौसम',
        'Monsoon': 'मानसून',
        
        # Buttons & Actions
        'Check for New Alerts': 'नई चेतावनियाँ देखें',
        'Refresh': 'रीफ्रेश',
        'Update': 'अपडेट',
        'Save': 'सेव',
        'Cancel': 'रद्द',
        'Submit': 'सबमिट',
        'Reset': 'रीसेट',
        'Clear': 'साफ़',
        'Search': 'खोजें',
        'Filter': 'फ़िल्टर',
        'Export': 'निर्यात',
        'Import': 'आयात',
        'Download': 'डाउनलोड',
        'Upload': 'अपलोड',
        'Print': 'प्रिंट',
        'Share': 'साझा करें',
        'Edit': 'संपादित करें',
        'Delete': 'हटाएं',
        'Add': 'जोड़ें',
        'Remove': 'हटाएं',
        'View Details': 'विवरण देखें',
        'Assess Credit Risk': 'क्रेडिट जोखिम का आकलन करें',
        'Generate Report': 'रिपोर्ट जेनरेट करें',
        'Analyze': 'विश्लेषण करें',
        'Approve': 'अनुमोदित करें',
        'Reject': 'अस्वीकार करें',
        'Review': 'समीक्षा करें',
        
        # Forms & Input Fields
        'Farmer Name': 'किसान का नाम',
        'Phone Number': 'फोन नंबर',
        'Address': 'पता',
        'Email': 'ईमेल',
        'Age': 'आयु',
        'Experience': 'अनुभव',
        'Education': 'शिक्षा',
        'Income': 'आय',
        'Loan Amount': 'ऋण राशि',
        'Loan Purpose': 'ऋण का उद्देश्य',
        'Repayment Period': 'चुकौती अवधि',
        'Collateral': 'संपार्श्विक',
        'Credit History': 'क्रेडिट इतिहास',
        'Bank Account': 'बैंक खाता',
        'PAN Number': 'पैन नंबर',
        'Aadhaar Number': 'आधार नंबर',
        'Location': 'स्थान',
        'State': 'राज्य',
        'District': 'जिला',
        'Village': 'गांव',
        'Pincode': 'पिनकोड',
        
        # Status & Messages
        'Last Check': 'अंतिम जाँच',
        'Last Updated': 'अंतिम अपडेट',
        'Status': 'स्थिति',
        'Active': 'सक्रिय',
        'Inactive': 'निष्क्रिय',
        'Pending': 'लंबित',
        'Approved': 'अनुमोदित',
        'Rejected': 'अस्वीकृत',
        'Under Review': 'समीक्षाधीन',
        'Completed': 'पूर्ण',
        'In Progress': 'प्रगतिशील',
        'Failed': 'असफल',
        'Success': 'सफल',
        'Error': 'त्रुटि',
        'Warning': 'चेतावनी',
        'Information': 'जानकारी',
        'Loading...': 'लोड हो रहा है...',
        'Please wait...': 'कृपया प्रतीक्षा करें...',
        'Processing...': 'प्रसंस्करण...',
        'Scanning weather conditions...': 'मौसमी परिस्थितियों की स्कैनिंग...',
        
        # Risk Levels & Severity
        'Low Risk': 'कम जोखिम',
        'Medium Risk': 'मध्यम जोखिम',
        'High Risk': 'उच्च जोखिम',
        'Very High Risk': 'अत्यधिक जोखिम',
        'Low': 'कम',
        'Medium': 'मध्यम',
        'High': 'उच्च',
        'Critical': 'गंभीर',
        'Severe': 'गंभीर',
        'Moderate': 'मध्यम',
        'Minimal': 'न्यूनतम',
        
        # Analytics & Reports
        'Data Analytics': 'डेटा विश्लेषण',
        'Trend Analysis': 'प्रवृत्ति विश्लेषण',
        'Predictive Analytics': 'भविष्यसूचक विश्लेषण',
        'Risk Analytics': 'जोखिम विश्लेषण',
        'Financial Analytics': 'वित्तीय विश्लेषण',
        'Performance Metrics': 'प्रदर्शन मैट्रिक्स',
        'Key Performance Indicators': 'मुख्य प्रदर्शन संकेतक',
        'Dashboard Overview': 'डैशबोर्ड अवलोकन',
        'Summary Report': 'सारांश रिपोर्ट',
        'Detailed Report': 'विस्तृत रिपोर्ट',
        'Monthly Report': 'मासिक रिपोर्ट',
        'Annual Report': 'वार्षिक रिपोर्ट',
        'Custom Report': 'कस्टम रिपोर्ट',
        
        # Time Periods
        'Today': 'आज',
        'Yesterday': 'कल',
        'This Week': 'इस सप्ताह',
        'Last Week': 'पिछला सप्ताह',
        'This Month': 'इस महीने',
        'Last Month': 'पिछला महीना',
        'This Year': 'इस साल',
        'Last Year': 'पिछला साल',
        'Last 30 Days': 'पिछले 30 दिन',
        'Last 90 Days': 'पिछले 90 दिन',
        'Custom Date Range': 'कस्टम तारीख सीमा',
        'From Date': 'तारीख से',
        'To Date': 'तारीख तक',
        
        # Alert Messages
        'No critical weather alerts at this time': 'इस समय कोई गंभीर मौसम चेतावनी नहीं',
        'Weather alert check failed': 'मौसम चेतावनी जांच असफल',
        'Unable to fetch weather data. Please check your internet connection.': 'मौसम डेटा प्राप्त करने में असमर्थ। कृपया इंटरनेट कनेक्शन जांचें।',
        'System is online': 'सिस्टम ऑनलाइन है',
        'System is offline': 'सिस्टम ऑफलाइन है',
        'Data updated successfully': 'डेटा सफलतापूर्वक अपडेट किया गया',
        'Error updating data': 'डेटा अपडेट करने में त्रुटि',
        'No data available': 'कोई डेटा उपलब्ध नहीं',
        'Insufficient data for analysis': 'विश्लेषण के लिए अपर्याप्त डेटा',
        
        # Configuration & Settings
        'General Settings': 'सामान्य सेटिंग्स',
        'User Preferences': 'उपयोगकर्ता प्राथमिकताएं',
        'Notification Settings': 'सूचना सेटिंग्स',
        'Alert Thresholds': 'अलर्ट सीमा',
        'API Configuration': 'एपीआई कॉन्फ़िगरेशन',
        'Database Settings': 'डेटाबेस सेटिंग्स',
        'Backup Settings': 'बैकअप सेटिंग्स',
        'Security Settings': 'सुरक्षा सेटिंग्स',
        'Privacy Settings': 'गोपनीयता सेटिंग्स',
        'Theme': 'थीम',
        'Dark Mode': 'डार्क मोड',
        'Light Mode': 'लाइट मोड',
    },
    
    'kn': {
        # Main Platform Headers
        'Capital One AgriCred AI Platform': 'ಕ್ಯಾಪಿಟಲ್ ವನ್ ಅಗ್ರಿಕ್ರೆಡ್ ಎಐ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್',
        'Advanced Agricultural Credit Intelligence & Risk Management': 'ಸುಧಾರಿತ ಕೃಷಿ ಸಾಲ ಬುದ್ಧಿವಂತಿಕೆ ಮತ್ತು ಅಪಾಯ ನಿರ್ವಹಣೆ',
        'Empowering financial institutions with AI-driven insights for agricultural lending': 'ಕೃಷಿ ಸಾಲಕ್ಕಾಗಿ AI-ಚಾಲಿತ ಒಳನೋಟಗಳೊಂದಿಗೆ ಹಣಕಾಸು ಸಂಸ್ಥೆಗಳಿಗೆ ಸಬಲೀಕರಣ',
        
        # Navigation & Menu Items
        'Executive Summary': 'ಕಾರ್ಯನಿರ್ವಾಹಕ ಸಾರಾಂಶ',
        'Portfolio Analytics': 'ಪೋರ್ಟ್‌ಫೋಲಿಯೋ ವಿಶ್ಲೇಷಣೆ',
        'Credit Risk Scoring': 'ಕ್ರೆಡಿಟ್ ಅಪಾಯ ಸ್ಕೋರಿಂಗ್',
        'Agentic AI Intelligence': 'ಏಜೆಂಟಿಕ್ AI ಬುದ್ಧಿವಂತಿಕೆ',
        'Weather Risk Monitor': 'ಹವಾಮಾನ ಅಪಾಯ ಮಾನಿಟರ್',
        'Market Intelligence': 'ಮಾರುಕಟ್ಟೆ ಬುದ್ಧಿವಂತಿಕೆ',
        'Performance Analytics': 'ಕಾರ್ಯಕ್ಷಮತೆ ವಿಶ್ಲೇಷಣೆ',
        'System Configuration': 'ಸಿಸ್ಟಮ್ ಸಂರಚನೆ',
        'Multi-lingual Demo': 'ಬಹುಭಾಷಾ ಡೆಮೊ',
        'Offline Capabilities': 'ಆಫ್‌ಲೈನ್ ಸಾಮರ್ಥ್ಯಗಳು',
        'Dashboard': 'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್',
        'Home': 'ಮನೆ',
        'Settings': 'ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Help': 'ಸಹಾಯ',
        'Profile': 'ಪ್ರೋಫೈಲ್',
        'Logout': 'ಲಾಗ್‌ಔಟ್',
        
        # Language & Accessibility
        'Select Language': 'ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'Current Language': 'ಪ್ರಸ್ತುತ ಭಾಷೆ',
        'Language': 'ಭಾಷೆ',
        'Accessibility': 'ಪ್ರವೇಶಿಸುವಿಕೆ',
        'Font Size': 'ಅಕ್ಷರದ ಗಾತ್ರ',
        'High Contrast Mode': 'ಹೆಚ್ಚಿನ ವ್ಯತ್ಯಾಸ ಮೋಡ್',
        
        # Dashboard Metrics & KPIs
        'Active Farmers': 'ಸಕ್ರಿಯ ಕೃಷಿಕರು',
        'Active Loans': 'ಸಕ್ರಿಯ ಸಾಲಗಳು',
        'High Risk Alerts': 'ಹೆಚ್ಚಿನ ಅಪಾಯದ ಎಚ್ಚರಿಕೆಗಳು',
        'Rainfall Alerts': 'ಮಳೆಯ ಎಚ್ಚರಿಕೆಗಳು',
        'Safe Conditions': 'ಸುರಕ್ಷಿತ ಸ್ಥಿತಿಗಳು',
        'Portfolio Value': 'ಪೋರ್ಟ್‌ಫೋಲಿಯೋ ಮೌಲ್ಯ',
        'Default Rate': 'ಡಿಫಾಲ್ಟ್ ದರ',
        'Avg Credit Score': 'ಸರಾಸರಿ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್',
        'Total Revenue': 'ಒಟ್ಟು ಆದಾಯ',
        'Net Profit': 'ನಿವ್ವಳ ಲಾಭ',
        'Cash Flow': 'ನಗದು ಹರಿವು',
        'Risk Assessment': 'ಅಪಾಯ ಮೌಲ್ಯಮಾಪನ',
        'Loan Approval Rate': 'ಸಾಲ ಅನುಮೋದನೆ ದರ',
        'NPAs': 'ಎನ್‌ಪಿಎ',
        'Interest Rate': 'ಬಡ್ಡಿ ದರ',
        'Collateral Value': 'ಮೂಢನಂಬಿಕೆ ಮೌಲ್ಯ',
        
        # Weather & Agricultural Terms
        'Live Weather Risk Monitoring System': 'ಲೈವ್ ಹವಾಮಾನ ಅಪಾಯ ಮೇಲ್ವಿಚಾರಣಾ ವ್ಯವಸ್ಥೆ',
        'Regional Weather Risk Map': 'ಪ್ರಾದೇಶಿಕ ಹವಾಮಾನ ಅಪಾಯ ನಕ್ಷೆ',
        'Recent Weather Activity': 'ಇತ್ತೀಚಿನ ಹವಾಮಾನ ಚಟುವಟಿಕೆ',
        'Temperature': 'ತಾಪಮಾನ',
        'Humidity': 'ಆರ್ದ್ರತೆ',
        'Wind Speed': 'ಗಾಳಿಯ ವೇಗ',
        'Rainfall': 'ಮಳೆಯ ಪ್ರಮಾಣ',
        'Soil Moisture': 'ಮಣ್ಣಿನ ತೇವಾಂಶ',
        'Crop Health': 'ಬೆಳೆಯ ಆರೋಗ್ಯ',
        'Irrigation': 'ನೀರಾವರಿ',
        'Harvest': 'ಸುಗ್ಗಿ',
        'Sowing': 'ಬಿತ್ತನೆ',
        'Fertilizer': 'ಗೊಬ್ಬರ',
        'Pesticide': 'ಕೀಟನಾಶಕ',
        'Yield': 'ಇಳುವರಿ',
        'Land Area': 'ಭೂಮಿ ಪ್ರದೇಶ',
        'Crop Type': 'ಬೆಳೆಯ ಪ್ರಕಾರ',
        'Season': 'ಋತು',
        'Monsoon': 'ಮುಂಗಾರು',
        
        # Buttons & Actions
        'Check for New Alerts': 'ಹೊಸ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ',
        'Refresh': 'ರಿಫ್ರೆಶ್',
        'Update': 'ಅಪ್‌ಡೇಟ್',
        'Save': 'ಉಳಿಸಿ',
        'Cancel': 'ರದ್ದುಮಾಡಿ',
        'Submit': 'ಸಲ್ಲಿಸಿ',
        'Reset': 'ಮರುಹೊಂದಿಸಿ',
        'Clear': 'ತೆರವುಗೊಳಿಸಿ',
        'Search': 'ಹುಡುಕಿ',
        'Filter': 'ಫಿಲ್ಟರ್',
        'Export': 'ರಫ್ತು',
        'Import': 'ಆಮದು',
        'Download': 'ಡೌನ್‌ಲೋಡ್',
        'Upload': 'ಅಪ್‌ಲೋಡ್',
        'Print': 'ಮುದ್ರಿಸಿ',
        'Share': 'ಹಂಚಿಕೊಳ್ಳಿ',
        'Edit': 'ಸಂಪಾದಿಸಿ',
        'Delete': 'ಅಳಿಸಿ',
        'Add': 'ಸೇರಿಸಿ',
        'Remove': 'ತೆಗೆದುಹಾಕಿ',
        'View Details': 'ವಿವರಗಳನ್ನು ವೀಕ್ಷಿಸಿ',
        'Assess Credit Risk': 'ಕ್ರೆಡಿಟ್ ಅಪಾಯವನ್ನು ನಿರ್ಣಯಿಸಿ',
        'Generate Report': 'ವರದಿ ಉತ್ಪಾದಿಸಿ',
        'Analyze': 'ವಿಶ್ಲೇಷಿಸಿ',
        'Approve': 'ಅನುಮೋದಿಸಿ',
        'Reject': 'ತಿರಸ್ಕರಿಸಿ',
        'Review': 'ಪರಿಶೀಲಿಸಿ',
        
        # Forms & Input Fields
        'Farmer Name': 'ಕೃಷಿಕರ ಹೆಸರು',
        'Phone Number': 'ಫೋನ್ ಸಂಖ್ಯೆ',
        'Address': 'ವಿಳಾಸ',
        'Email': 'ಇಮೇಲ್',
        'Age': 'ವಯಸ್ಸು',
        'Experience': 'ಅನುಭವ',
        'Education': 'ಶಿಕ್ಷಣ',
        'Income': 'ಆದಾಯ',
        'Loan Amount': 'ಸಾಲದ ಮೊತ್ತ',
        'Loan Purpose': 'ಸಾಲದ ಉದ್ದೇಶ',
        'Repayment Period': 'ಮರುಪಾವತಿ ಅವಧಿ',
        'Collateral': 'ಮೂಢನಂಬಿಕೆ',
        'Credit History': 'ಕ್ರೆಡಿಟ್ ಇತಿಹಾಸ',
        'Bank Account': 'ಬ್ಯಾಂಕ್ ಖಾತೆ',
        'PAN Number': 'ಪ್ಯಾನ್ ಸಂಖ್ಯೆ',
        'Aadhaar Number': 'ಆಧಾರ್ ಸಂಖ್ಯೆ',
        'Location': 'ಸ್ಥಳ',
        'State': 'ರಾಜ್ಯ',
        'District': 'ಜಿಲ್ಲೆ',
        'Village': 'ಗ್ರಾಮ',
        'Pincode': 'ಪಿನ್‌ಕೋಡ್',
        
        # Status & Messages
        'Last Check': 'ಕೊನೆಯ ಪರಿಶೀಲನೆ',
        'Last Updated': 'ಕೊನೆಯ ಅಪ್‌ಡೇಟ್',
        'Status': 'ಸ್ಥಿತಿ',
        'Active': 'ಸಕ್ರಿಯ',
        'Inactive': 'ನಿಷ್ಕ್ರಿಯ',
        'Pending': 'ಬಾಕಿ',
        'Approved': 'ಅನುಮೋದಿತ',
        'Rejected': 'ತಿರಸ್ಕರಿಸಲಾಗಿದೆ',
        'Under Review': 'ಪರಿಶೀಲನೆಯಲ್ಲಿ',
        'Completed': 'ಪೂರ್ಣಗೊಂಡಿದೆ',
        'In Progress': 'ಪ್ರಗತಿಯಲ್ಲಿದೆ',
        'Failed': 'ವಿಫಲವಾಗಿದೆ',
        'Success': 'ಯಶಸ್ವಿ',
        'Error': 'ದೋಷ',
        'Warning': 'ಎಚ್ಚರಿಕೆ',
        'Information': 'ಮಾಹಿತಿ',
        'Loading...': 'ಲೋಡ್ ಆಗುತ್ತಿದೆ...',
        'Please wait...': 'ದಯವಿಟ್ಟು ಕಾಯಿರಿ...',
        'Processing...': 'ಪ್ರಕ್ರಿಯೆಗೊಳಿಸುತ್ತಿದೆ...',
        'Scanning weather conditions...': 'ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡುತ್ತಿದೆ...',
        
        # Risk Levels & Severity
        'Low Risk': 'ಕಡಿಮೆ ಅಪಾಯ',
        'Medium Risk': 'ಮಧ್ಯಮ ಅಪಾಯ',
        'High Risk': 'ಹೆಚ್ಚಿನ ಅಪಾಯ',
        'Very High Risk': 'ಅತ್ಯಧಿಕ ಅಪಾಯ',
        'Low': 'ಕಡಿಮೆ',
        'Medium': 'ಮಧ್ಯಮ',
        'High': 'ಹೆಚ್ಚಿನ',
        'Critical': 'ನಿರ್ಣಾಯಕ',
        'Severe': 'ತೀವ್ರ',
        'Moderate': 'ಮಧ್ಯಮ',
        'Minimal': 'ಕನಿಷ್ಠ',
        
        # Analytics & Reports
        'Data Analytics': 'ಡೇಟಾ ವಿಶ್ಲೇಷಣೆ',
        'Trend Analysis': 'ಪ್ರವೃತ್ತಿ ವಿಶ್ಲೇಷಣೆ',
        'Predictive Analytics': 'ಭವಿಷ್ಯಸೂಚಕ ವಿಶ್ಲೇಷಣೆ',
        'Risk Analytics': 'ಅಪಾಯ ವಿಶ್ಲೇಷಣೆ',
        'Financial Analytics': 'ಹಣಕಾಸು ವಿಶ್ಲೇಷಣೆ',
        'Performance Metrics': 'ಕಾರ್ಯಕ್ಷಮತೆ ಮಾಪಕಗಳು',
        'Key Performance Indicators': 'ಮುಖ್ಯ ಕಾರ್ಯಕ್ಷಮತೆ ಸೂಚಕಗಳು',
        'Dashboard Overview': 'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ಅವಲೋಕನ',
        'Summary Report': 'ಸಾರಾಂಶ ವರದಿ',
        'Detailed Report': 'ವಿವರವಾದ ವರದಿ',
        'Monthly Report': 'ಮಾಸಿಕ ವರದಿ',
        'Annual Report': 'ವಾರ್ಷಿಕ ವರದಿ',
        'Custom Report': 'ಕಸ್ಟಮ್ ವರದಿ',
        
        # Time Periods
        'Today': 'ಇಂದು',
        'Yesterday': 'ನಿನ್ನೆ',
        'This Week': 'ಈ ವಾರ',
        'Last Week': 'ಕಳೆದ ವಾರ',
        'This Month': 'ಈ ತಿಂಗಳು',
        'Last Month': 'ಕಳೆದ ತಿಂಗಳು',
        'This Year': 'ಈ ವರ್ಷ',
        'Last Year': 'ಕಳೆದ ವರ್ಷ',
        'Last 30 Days': 'ಕಳೆದ 30 ದಿನಗಳು',
        'Last 90 Days': 'ಕಳೆದ 90 ದಿನಗಳು',
        'Custom Date Range': 'ಕಸ್ಟಮ್ ದಿನಾಂಕ ಶ್ರೇಣಿ',
        'From Date': 'ದಿನಾಂಕದಿಂದ',
        'To Date': 'ದಿನಾಂಕಕ್ಕೆ',
        
        # Alert Messages
        'No critical weather alerts at this time': 'ಈ ಸಮಯದಲ್ಲಿ ಯಾವುದೇ ಗಂಭೀರ ಹವಾಮಾನ ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ',
        'Weather alert check failed': 'ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ',
        'Unable to fetch weather data. Please check your internet connection.': 'ಹವಾಮಾನ ಡೇಟಾವನ್ನು ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕವನ್ನು ಪರಿಶೀಲಿಸಿ.',
        'System is online': 'ಸಿಸ್ಟಮ್ ಆನ್‌ಲೈನ್‌ನಲ್ಲಿದೆ',
        'System is offline': 'ಸಿಸ್ಟಮ್ ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿದೆ',
        'Data updated successfully': 'ಡೇಟಾ ಯಶಸ್ವಿಯಾಗಿ ನವೀಕರಿಸಲಾಗಿದೆ',
        'Error updating data': 'ಡೇಟಾ ನವೀಕರಿಸುವಲ್ಲಿ ದೋಷ',
        'No data available': 'ಯಾವುದೇ ಡೇಟಾ ಲಭ್ಯವಿಲ್ಲ',
        'Insufficient data for analysis': 'ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಅಸಮರ್ಪಕ ಡೇಟಾ',
        
        # Configuration & Settings
        'General Settings': 'ಸಾಮಾನ್ಯ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'User Preferences': 'ಬಳಕೆದಾರರ ಆದ್ಯತೆಗಳು',
        'Notification Settings': 'ಅಧಿಸೂಚನೆ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Alert Thresholds': 'ಎಚ್ಚರಿಕೆ ಮಿತಿಗಳು',
        'API Configuration': 'API ಸಂರಚನೆ',
        'Database Settings': 'ಡೇಟಾಬೇಸ್ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Backup Settings': 'ಬ್ಯಾಕಪ್ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Security Settings': 'ಭದ್ರತಾ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Privacy Settings': 'ಗೌಪ್ಯತೆ ಸೆಟ್ಟಿಂಗ್‌ಗಳು',
        'Theme': 'ಥೀಮ್',
        'Dark Mode': 'ಡಾರ್ಕ್ ಮೋಡ್',
        'Light Mode': 'ಲೈಟ್ ಮೋಡ್',
    }
}

def translate_text(text: str, target_lang: str = 'en') -> str:
    """
    Translate UI text or arbitrary text into the target language.
    1. First, check our built-in _UI_TRANSLATIONS for exact matches.
    2. If not found and googletrans is available, call the API for translation.
    3. Otherwise, return the original text.
    """
    # 1. Built-in UI translation
    ui_map = _UI_TRANSLATIONS.get(target_lang, {})
    if text in ui_map:
        return ui_map[text]
    
    # 2. Use googletrans for arbitrary text (good for longer context)
    if _translator:
        try:
            result = _translator.translate(text, dest=target_lang)
            return result.text
        except Exception as e:
            logger.warning(f"External translation failed for '{text}' to '{target_lang}': {e}")
    
    # 3. Fallback to original
    return text



class MultiLingualMultiModalManager:
    """Manages multi-lingual and multi-modal capabilities"""
    
    def __init__(self):
        self.languages = self._initialize_language_support()
        self.recognizer = sr.Recognizer()
        self.tts_engine = None
        self._initialize_tts()
        self.query_history = []
        
    def _initialize_language_support(self) -> Dict[str, LanguageSupport]:
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
            confidence_threshold=0.7,
            fallback_language='en'
        )
        
        # Marathi (Maharashtra)
        languages['mr'] = LanguageSupport(
            code='mr',
            name='Marathi',
            native_name='मराठी',
            tts_voice='marathi',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Bengali (West Bengal)
        languages['bn'] = LanguageSupport(
            code='bn',
            name='Bengali',
            native_name='বাংলা',
            tts_voice='bengali',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Telugu (Andhra Pradesh, Telangana)
        languages['te'] = LanguageSupport(
            code='te',
            name='Telugu',
            native_name='తెలుగు',
            tts_voice='telugu',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Tamil (Tamil Nadu)
        languages['ta'] = LanguageSupport(
            code='ta',
            name='Tamil',
            native_name='தமிழ்',
            tts_voice='tamil',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Gujarati (Gujarat)
        languages['gu'] = LanguageSupport(
            code='gu',
            name='Gujarati',
            native_name='ગુજરાતી',
            tts_voice='gujarati',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Punjabi (Punjab)
        languages['pa'] = LanguageSupport(
            code='pa',
            name='Punjabi',
            native_name='ਪੰਜਾਬੀ',
            tts_voice='punjabi',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Kannada (Karnataka)
        languages['kn'] = LanguageSupport(
            code='kn',
            name='Kannada',
            native_name='ಕನ್ನಡ',
            tts_voice='kannada',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        # Malayalam (Kerala)
        languages['ml'] = LanguageSupport(
            code='ml',
            name='Malayalam',
            native_name='മലയാളം',
            tts_voice='malayalam',
            confidence_threshold=0.6,
            fallback_language='hi'
        )
        
        return languages
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            
            # Set default voice
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None
    
    def detect_language(self, text: str) -> Tuple[str, float]:
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
    
    def transcribe_audio(self, audio_data: bytes, language: str = 'en') -> Tuple[str, float]:
        """Transcribe audio data to text"""
        try:
            # Convert audio data to AudioData object
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Try to transcribe with specified language
            if language in self.languages:
                try:
                    text = self.recognizer.recognize_google(
                        audio, 
                        language=language,
                        show_all=False
                    )
                    return text, 0.9
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            
            # Fallback to English
            try:
                text = self.recognizer.recognize_google(audio, language='en-US')
                return text, 0.7
            except sr.UnknownValueError:
                return "", 0.0
            except sr.RequestError:
                return "", 0.0
                
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return "", 0.0
    
    def process_voice_input(self, audio_data: bytes, sample_rate: int = 16000) -> VoiceInput:
        """Process voice input and return structured data"""
        # Calculate audio duration
        duration = len(audio_data) / (sample_rate * 2)  # Assuming 16-bit audio
        
        # Create voice input object
        voice_input = VoiceInput(
            audio_data=audio_data,
            language_detected='en',  # Will be updated after transcription
            confidence=0.0,
            duration_seconds=duration,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        # Transcribe audio
        transcription, confidence = self.transcribe_audio(audio_data)
        voice_input.transcription = transcription
        voice_input.transcription_confidence = confidence
        
        if transcription:
            # Detect language from transcription
            detected_lang, lang_confidence = self.detect_language(transcription)
            voice_input.language_detected = detected_lang
            voice_input.confidence = lang_confidence
        
        return voice_input
    
    def create_multimodal_query(self, text: str = None, voice: VoiceInput = None) -> MultiModalQuery:
        """Create a multi-modal query from text and/or voice input"""
        query_id = hashlib.md5(f"{pd.Timestamp.now().isoformat()}".encode()).hexdigest()[:8]
        
        if text and voice:
            input_type = "mixed"
            detected_lang, confidence = self.detect_language(text)
        elif text:
            input_type = "text"
            detected_lang, confidence = self.detect_language(text)
        elif voice:
            input_type = "voice"
            detected_lang = voice.language_detected
            confidence = voice.confidence
        else:
            raise ValueError("Either text or voice input must be provided")
        
        query = MultiModalQuery(
            query_id=query_id,
            input_type=input_type,
            text_content=text,
            voice_content=voice,
            detected_language=detected_lang,
            confidence=confidence,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        # Store in history
        self.query_history.append(query)
        
        return query
    
    def extract_intent_and_entities(self, query: MultiModalQuery) -> Tuple[str, List[Dict]]:
        """Extract intent and entities from query"""
        text = query.text_content or (query.voice_content.transcription if query.voice_content else "")
        if not text:
            return "unknown", []
        
        text_lower = text.lower()
        
        # Intent detection based on keywords
        intents = {
            "credit_inquiry": ["credit", "loan", "borrow", "money", "finance", "कर्ज", "ऋण", "धन"],
            "weather_info": ["weather", "rain", "temperature", "मौसम", "बारिश", "तापमान"],
            "market_price": ["price", "market", "mandi", "crop", "फसल", "मंडी", "कीमत"],
            "soil_health": ["soil", "fertilizer", "मिट्टी", "खाद", "उर्वरक"],
            "government_scheme": ["scheme", "subsidy", "government", "योजना", "सब्सिडी", "सरकार"],
            "crop_advice": ["crop", "plant", "harvest", "फसल", "बोना", "काटना"],
            "pest_control": ["pest", "disease", "insect", "कीट", "रोग", "कीड़ा"]
        }
        
        detected_intent = "general_inquiry"
        max_score = 0
        
        for intent, keywords in intents.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > max_score:
                max_score = score
                detected_intent = intent
        
        # Entity extraction
        entities = []
        
        # Crop entities
        crops = ["wheat", "rice", "cotton", "sugarcane", "soybean", "maize", "गेहूं", "चावल", "कपास", "गन्ना"]
        for crop in crops:
            if crop in text_lower:
                entities.append({"type": "crop", "value": crop, "confidence": 0.9})
        
        # Location entities
        locations = ["punjab", "maharashtra", "up", "karnataka", "gujarat", "पंजाब", "महाराष्ट्र", "उत्तर प्रदेश"]
        for location in locations:
            if location in text_lower:
                entities.append({"type": "location", "value": location, "confidence": 0.9})
        
        # Time entities
        time_keywords = ["today", "tomorrow", "week", "month", "आज", "कल", "सप्ताह", "महीना"]
        for time_word in time_keywords:
            if time_word in text_lower:
                entities.append({"type": "time", "value": time_word, "confidence": 0.8})
        
        return detected_intent, entities
    
    def text_to_speech(self, text: str, language: str = 'en', output_format: str = 'audio') -> bytes:
        """Convert text to speech"""
        try:
            if language in ['hi', 'mr', 'bn', 'te', 'ta', 'gu', 'pa', 'kn', 'ml']:
                # Use gTTS for Indian languages
                tts = gTTS(text=text, lang=language, slow=False)
                
                if output_format == 'audio':
                    # Return audio data
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    return audio_buffer.read()
                else:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tts.save(temp_file.name)
                    return temp_file.name.encode()
            else:
                # Use pyttsx3 for English
                if self.tts_engine:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    self.tts_engine.save_to_file(text, temp_file.name)
                    self.tts_engine.runAndWait()
                    return temp_file.name.encode()
                else:
                    # Fallback to gTTS
                    tts = gTTS(text=text, lang='en', slow=False)
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    return audio_buffer.read()
                    
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return b""
    
    def create_sms_text(self, response: str, language: str = 'en') -> str:
        """Create SMS-friendly text from response"""
        # Remove HTML tags and special characters
        import re
        clean_text = re.sub(r'<[^>]+>', '', response)
        clean_text = re.sub(r'[^\w\s\.\,\!\?\-]', '', clean_text)
        
        # Truncate if too long for SMS
        max_length = 160
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length-3] + "..."
        
        return clean_text
    
    def get_language_display_name(self, language_code: str) -> str:
        """Get display name for language code"""
        if language_code in self.languages:
            return self.languages[language_code].native_name
        return language_code.upper()
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {
                "code": code,
                "name": lang.name,
                "native_name": lang.native_name
            }
            for code, lang in self.languages.items()
        ]
    
    def export_query_log(self, format: str = 'csv') -> str:
        """Export query history log"""
        if not self.query_history:
            return ""
        
        if format == 'csv':
            # Convert to DataFrame and export
            data = []
            for query in self.query_history:
                data.append({
                    'query_id': query.query_id,
                    'timestamp': query.timestamp,
                    'input_type': query.input_type,
                    'detected_language': query.detected_language,
                    'confidence': query.confidence,
                    'text_content': query.text_content or '',
                    'voice_transcription': query.voice_content.transcription if query.voice_content else '',
                    'intent': query.intent or '',
                    'entities': str(query.entities) if query.entities else ''
                })
            
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        
        elif format == 'json':
            # Export as JSON
            data = []
            for query in self.query_history:
                query_dict = {
                    'query_id': query.query_id,
                    'timestamp': query.timestamp,
                    'input_type': query.input_type,
                    'detected_language': query.detected_language,
                    'confidence': query.confidence,
                    'text_content': query.text_content,
                    'voice_content': {
                        'transcription': query.voice_content.transcription if query.voice_content else None,
                        'confidence': query.voice_content.transcription_confidence if query.voice_content else None
                    } if query.voice_content else None,
                    'intent': query.intent,
                    'entities': query.entities
                }
                data.append(query_dict)
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

# Global instance
multimodal_manager = MultiLingualMultiModalManager()
