# Regimeaware-RL-Portfolio-optimization
Regime-aware reinforcement learning framework for multi-asset portfolio optimization using PPO, Hidden Markov Models, risk-aware reward shaping, and explainable AI.
## Overview

This project implements a regime-aware reinforcement learning framework for portfolio optimisation using:

- Hidden Markov Models (HMM)
- PPO Reinforcement Learning
- Risk-aware reward shaping
- SHAP explainability
- Streamlit decision intelligence dashboard

The framework models changing market regimes such as:

- Bull markets
- Bear markets
- Crisis periods

and dynamically adjusts portfolio allocations.

---

# Features

- Regime detection using Gaussian HMM
- PPO reinforcement learning agent
- Risk-aware reward engineering
- Portfolio backtesting
- Stress-period evaluation
- SHAP explainability analysis
- Interactive Streamlit dashboard
- Animated portfolio simulation

---

# Project Structure

```text
project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dashboard/
│   └── streamlit_animated_simulator.py
│
├── outputs/
├── data/
└── src/
```

---

# Installation

## Clone Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Main Pipeline

```bash
python main.py
```

This generates:

- Backtest results
- Portfolio metrics
- SHAP explainability outputs
- Dashboard CSV files
- Visualisations

Outputs are saved inside:

```text
outputs/
```

---

# Running the Dashboard

```bash
streamlit run dashboard/streamlit_animated_simulator.py
```

---

# Dashboard Features

- Portfolio growth animation
- Regime probability tracking
- Drawdown analysis
- Asset allocation visualisation
- Decision intelligence explanations
- Interactive simulation controls

---

# Technologies Used

- Python
- Streamlit
- Plotly
- Stable-Baselines3
- Gymnasium
- HMMlearn
- SHAP
- Pandas
- NumPy
- Scikit-learn

---

# Research Contribution

This framework investigates whether regime-aware reinforcement learning can improve:

- Risk-adjusted returns
- Downside-risk control
- Portfolio adaptability during market stress

compared with traditional portfolio strategies.

---

# Author

Suresh A

MSc Data Science
