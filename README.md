# AgriCredAI - Agricultural Credit Intelligence Platform

<div align="center">
  <h3>Data-Driven Credit Risk Assessment & Financial Advisory</h3>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python"/>
    </a>
    <a href="https://streamlit.io/">
      <img src="https://img.shields.io/badge/Streamlit-1.28+-red" alt="Streamlit"/>
    </a>
    <a href="https://xgboost.readthedocs.io/">
      <img src="https://img.shields.io/badge/Model-XGBoost%20v3.0-orange" alt="XGBoost"/>
    </a>
  </p>
</div>

---

## 🚀 Project Overview
**AgriCredAI** is a comprehensive FinTech platform designed to modernize credit risk assessment for the agricultural sector. It utilizes **XGBoost machine learning** models and a **Expert Intelligence System** to provide real-time credit scoring and financial recommendations.

By integrating live data from **OpenWeatherMap** (for weather risk) and **Agmarknet** (for market pricing), the platform ensures that credit evaluations are responsive to current environmental and economic conditions.

---

## 🏗️ Technical Architecture

### 1. Hybrid Data Pipeline
The system utilizes a custom pipeline (`AgroScoreInferencePipeline`) that handles:
- **Synthetic Data Seeding**: Generates realistic farmer profiles for local testing.
- **Live API Integration**: Polls real-time weather and market data to adjust risk factors.
- **Fault Tolerance**: Implements retry logic and fallback data for external API dependencies.

### 2. Expert Intelligence Advisory
The platform features a multi-advisor framework coordinated by an **IntelOrchestrator**:
- **Dynamic Financing Advisor**: Recommends loan structures and repayment schedules based on crop harvest cycles.
- **Market Advisory Advisor**: Analyzes commodity price trends for trading intelligence.
- **Carbon Credit Advisor**: Tracks sustainable farming practices and tokenizes credits with blockchain-simulated hashes.

---

## 📐 The Formula Base

The system's logic is defined by the following quantitative frameworks:

### I. Risk Score Index ($R_s$)
Aggregates normalized features across weather, market, and credit history:
$$R_s = \sum_{i=1}^{n} (Feature_i \times Weight_i)$$

### II. Credit Score Calibration ($C$)
Normalizes the ML model's probability of default into an industry-standard range:
$$Score = Base + \text{Factor} \times \ln\left(\frac{P(\text{Good})}{P(\text{Default})}\right)$$

### III. Loan Capacity ($L_c$)
Calculates maximum loan eligibility based on projected income and risk mitigation:
$$L_c = (\text{Income}_{\text{est}} \times \text{Multiplier}) \times (1 - R_s)$$

---

## 🧠 Machine Learning & Explainability
- **The Model:** XGBoost implementation processing 79 unique feature dimensions.
- **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** to provide transparent reasoning for each credit card score, detailing exactly which factors influenced the decision.

---

## 📂 Project Structure
```
AgriCredAI/
├── 📄 advanced_app.py                   # Main Streamlit dashboard
├── 🧠 advanced_data_pipeline.py         # Hybrid data & inference pipeline
├── 🎯 advanced_ml_model.py              # XGBoost ML model implementation
├── 🤖 agri_intel_core.py                # Expert intelligence core framework
├── 💰 dynamic_financing_advisor.py      # Financing and loan advisory
├── 📊 market_advisory_advisor.py        # Market and pricing advisory
├── 🌿 carbon_credit_advisor.py          # Sustainable practice tracking
├── 🎯 explainable_ai_core.py            # SHAP-based model explainability
├── 📋 requirements.txt                  # Dependency list
└── 📖 README.md                         # Documentation
```

---

## 🚀 Getting Started

1. **Installation:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup:**
   Configure your `WEATHER_API_KEY` in the `.env` file.

3. **Run Application:**
   ```bash
   streamlit run advanced_app.py
   ```
