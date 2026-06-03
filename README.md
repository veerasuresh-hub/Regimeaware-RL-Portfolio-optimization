# Regime-Aware Deep Reinforcement Learning for Interpretable Multi-Asset Portfolio Optimization

Regime-aware reinforcement learning framework for multi-asset portfolio optimisation using PPO, Hidden Markov Models, risk-aware reward shaping, and explainable AI.

## Overview

This project implements a regime-aware reinforcement learning framework for portfolio optimisation. The framework combines:

* Hidden Markov Models (HMM)
* PPO reinforcement learning
* Risk-aware reward shaping
* Portfolio backtesting
* Stress-period evaluation
* SHAP explainability
* Streamlit dashboard visualisation

The system identifies changing market conditions such as Bull, Bear, and Crisis regimes, and dynamically adjusts portfolio allocation decisions based on evolving market behaviour.

---

# Features

* Regime detection using Gaussian HMM
* PPO reinforcement learning agent
* Risk-aware reward engineering
* Portfolio backtesting
* Stress-period evaluation
* SHAP explainability analysis
* Interactive Streamlit dashboard
* Animated portfolio simulation

---

# Project Structure

```text
Regimeaware-RL-Portfolio-optimization/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dashboard/
│   └── streamlit_animated_simulator.py
│
├── data/
│   └── cleaned_real_asset_returns.csv
│
├── outputs/
│   ├── final_dashboard_decision_table.csv
│   ├── strategy_results.csv
│   ├── shap_feature_importance.csv
│   └── other generated outputs
│
└── src/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/veerasuresh-hub/Regimeaware-RL-Portfolio-optimization.git
```

## Move Into Project Folder

```bash
cd Regimeaware-RL-Portfolio-optimization
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Main Pipeline

Run the full research pipeline using:

```bash
python main.py
```

The pipeline performs:

* Data loading and preprocessing
* Feature engineering
* HMM regime detection
* PPO reinforcement learning training
* Reinforcement learning backtesting
* Baseline strategy comparison
* Stress testing
* Hypothesis testing
* SHAP explainability analysis
* Dashboard export generation

Generated outputs are automatically stored inside:

```text
outputs/
```

---

# Running the Streamlit Dashboard

The dashboard reads the exported file:

```text
outputs/final_dashboard_decision_table.csv
```

Launch the dashboard using:

```bash
streamlit run dashboard/streamlit_animated_simulator.py
```

After execution, Streamlit opens the dashboard in the browser using:

```text
http://localhost:8501
```

---

# Dashboard Features

* Portfolio growth animation
* Regime probability tracking
* Drawdown analysis
* Asset allocation visualisation
* Decision intelligence explanations
* Interactive simulation controls

---

# Dataset

Dataset used in this project:

```text
data/cleaned_real_asset_returns.csv
```

---

# Outputs

The outputs folder contains:

* Portfolio performance metrics
* Reinforcement learning backtesting results
* Regime probability outputs
* Stress testing outputs
* SHAP explainability outputs
* Ablation study results
* Dashboard decision tables
* Visualisation files

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* HMMlearn
* Stable-Baselines3
* Gymnasium
* SHAP
* Plotly
* Streamlit

---

# Research Contribution

This project investigates whether combining regime detection, risk-aware reward shaping, and reinforcement learning can improve adaptive portfolio allocation and downside-risk control under changing market conditions.

---

# Author

Veera Suresh A
MSc Data Science
