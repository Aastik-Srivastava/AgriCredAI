#!/usr/bin/env python3
"""
Setup Script for Refactored AgriCredAI System
Helps configure and test the new modules
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version)
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'requests', 'sqlite3',
        'plotly', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# AgriCredAI Environment Configuration\n")
            f.write("# Add your API keys here\n\n")
            f.write("WEATHER_API_KEY=your_openweathermap_key_here\n")
            f.write("MARKET_API_KEY=your_agmarknet_key_here\n")
            f.write("SOIL_HEALTH_API_KEY=your_soil_api_key_here\n")
            f.write("DEBUG=True\n")
            f.write("TESTING=True\n")
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    # Create necessary directories
    directories = ['data_cache', 'offline_cache', 'logs', 'exports']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_modules():
    """Test if all new modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    modules_to_test = [
        'public_data_integration',
        'multilingual_multimodal', 
        'explainable_ai_core',
        'offline_edge_support',
        'demo_user_journey'
    ]
    
    failed_modules = []
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import: {', '.join(failed_modules)}")
        return False
    
    return True

def test_data_sources():
    """Test data source connectivity"""
    print("\n🌐 Testing data sources...")
    
    try:
        from public_data_integration import data_manager
        
        # Test connectivity
        is_online = data_manager.check_connectivity()
        if is_online:
            print("✅ Internet connectivity available")
        else:
            print("⚠️  No internet connectivity - will use offline mode")
        
        # Test data source initialization
        sources = data_manager.sources
        print(f"✅ Initialized {len(sources)} data sources")
        
        for name, source in sources.items():
            print(f"  - {source.name}: {source.data_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data source test failed: {e}")
        return False

def test_multilingual():
    """Test multi-lingual capabilities"""
    print("\n🗣️ Testing multi-lingual support...")
    
    try:
        from multilingual_multimodal import multimodal_manager
        
        # Test language support
        languages = multimodal_manager.get_supported_languages()
        print(f"✅ Supported languages: {len(languages)}")
        
        # Test language detection
        test_texts = [
            ("Hello world", "en"),
            ("नमस्ते दुनिया", "hi"),
            ("வணக்கம் உலகம்", "ta")
        ]
        
        for text, expected_lang in test_texts:
            detected_lang, confidence = multimodal_manager.detect_language(text)
            status = "✅" if detected_lang == expected_lang else "⚠️"
            print(f"  {status} '{text}' -> {detected_lang} (confidence: {confidence:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-lingual test failed: {e}")
        return False

def test_explainable_ai():
    """Test explainable AI capabilities"""
    print("\n🧠 Testing explainable AI...")
    
    try:
        from explainable_ai_core import explainable_ai
        
        # Test confidence scoring
        confidence = explainable_ai.calculate_confidence_score(
            data_quality=0.8,
            model_performance=0.9,
            feature_completeness=0.85,
            temporal_relevance=0.9,
            spatial_coverage=0.8
        )
        
        print(f"✅ Confidence scoring: {confidence.overall_confidence:.1%}")
        print(f"  - Data quality: {confidence.data_quality_confidence:.1%}")
        print(f"  - Model performance: {confidence.model_confidence:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Explainable AI test failed: {e}")
        return False

def test_offline_support():
    """Test offline support capabilities"""
    print("\n📱 Testing offline support...")
    
    try:
        from offline_edge_support import offline_support
        
        # Test database setup
        db_path = offline_support.db_path
        if db_path.exists():
            print(f"✅ Offline database: {db_path}")
        else:
            print(f"❌ Offline database not found: {db_path}")
            return False
        
        # Test data caching
        test_data = {"test": "data", "timestamp": "2024-01-01"}
        cache_hash = offline_support.cache_data(
            "test_cache", test_data, "test_source", 0.9, 1
        )
        
        if cache_hash:
            print("✅ Data caching working")
            
            # Test retrieval
            retrieved = offline_support.get_cached_data("test_cache", max_age_hours=2)
            if retrieved:
                print("✅ Data retrieval working")
            else:
                print("❌ Data retrieval failed")
                return False
        else:
            print("❌ Data caching failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Offline support test failed: {e}")
        return False

def test_demo_system():
    """Test demo system capabilities"""
    print("\n🎭 Testing demo system...")
    
    try:
        from demo_user_journey import demo_manager
        
        # Test scenario initialization
        scenarios = demo_manager.demo_scenarios
        print(f"✅ Demo scenarios: {len(scenarios)}")
        
        for scenario_id, scenario in scenarios.items():
            print(f"  - {scenario.title} ({scenario.difficulty_level})")
        
        # Test demo creation
        demo_id = demo_manager.start_demo("farmer_basic", "test_user")
        if demo_id:
            print(f"✅ Demo creation: {demo_id}")
            
            # Test progress tracking
            progress = demo_manager.get_demo_progress()
            if progress['status'] == 'active':
                print("✅ Progress tracking working")
            else:
                print("❌ Progress tracking failed")
                return False
        else:
            print("❌ Demo creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Demo system test failed: {e}")
        return False

def run_integration_test():
    """Run the integration test"""
    print("\n🔗 Running integration test...")
    
    try:
        # Test if integration example can be imported
        from integration_example import main
        print("✅ Integration example imported successfully")
        
        # Note: The actual Streamlit app would need to be run separately
        print("ℹ️  To run the full integration demo:")
        print("   streamlit run integration_example.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AgriCredAI Refactored System Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Test modules
    if not test_modules():
        print("\n❌ Module tests failed")
        sys.exit(1)
    
    # Test individual components
    tests = [
        ("Data Sources", test_data_sources),
        ("Multi-lingual", test_multilingual),
        ("Explainable AI", test_explainable_ai),
        ("Offline Support", test_offline_support),
        ("Demo System", test_demo_system),
        ("Integration", run_integration_test)
    ]
    
    print("\n🧪 Running component tests...")
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            failed_tests.append(test_name)
    
    # Summary
    print("\n" + "=" * 50)
    if not failed_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📱 To start using the system:")
        print("1. Set your API keys in the .env file")
        print("2. Run: streamlit run integration_example.py")
        print("3. Or run: streamlit run advanced_app.py")
    else:
        print(f"⚠️  {len(failed_tests)} tests failed: {', '.join(failed_tests)}")
        print("Please check the errors above and fix them.")
    
    print("\n📚 For more information, see REFACTORING_SUMMARY.md")

if __name__ == "__main__":
    main()
