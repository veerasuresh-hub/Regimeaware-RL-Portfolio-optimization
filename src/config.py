from dataclasses import dataclass

SEED = 42
OUTPUT_DIR = "outputs"


@dataclass
class Config:
    file_path: str = "data/cleaned_real_asset_returns.csv"

    train_ratio: float = 0.70

    rolling_window: int = 3
    include_drawdown_features: bool = True
    include_correlation_stress: bool = True

    n_regimes: int = 3
    hmm_iter: int = 500

    total_timesteps: int = 35_000

    transaction_cost: float = 0.0005
    volatility_penalty: float = 0.03
    crisis_penalty: float = 0.15
    reward_clip: float = 0.05
    turnover_penalty_weight: float = 0.005

    markowitz_window: int = 5
    risk_parity_window: int = 5

    shap_background_size: int = 20
    shap_sample_size: int = 20


CFG = Config()
