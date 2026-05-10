import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_drawdown_features(returns_df):
    wealth = (1.0 + returns_df).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / (peak + 1e-12)
    return drawdown.add_suffix("_DD")


def compute_correlation_stress_features(returns_df, rolling_window=3):
    out = pd.DataFrame(index=returns_df.index)
    avg_corr, max_corr = [], []

    for i in range(len(returns_df)):
        start = max(0, i - rolling_window + 1)
        window = returns_df.iloc[start:i + 1]

        if len(window) < 3 or window.shape[1] < 2:
            avg_corr.append(np.nan)
            max_corr.append(np.nan)
            continue

        corr = window.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]
        upper = upper[np.isfinite(upper)]

        if len(upper) == 0:
            avg_corr.append(np.nan)
            max_corr.append(np.nan)
        else:
            avg_corr.append(float(np.mean(upper)))
            max_corr.append(float(np.max(upper)))

    out["Avg_Corr_Stress"] = avg_corr
    out["Max_Corr_Stress"] = max_corr

    return out


def build_yearly_features(
    returns_df,
    rolling_window=3,
    include_drawdown_features=True,
    include_correlation_stress=True,
):
    returns = returns_df.copy()

    rolling_vol = returns.rolling(
        rolling_window,
        min_periods=2
    ).std().add_suffix("_Vol")

    rolling_momentum = returns.rolling(
        rolling_window,
        min_periods=2
    ).mean().add_suffix("_Mom")

    market_features = pd.DataFrame(index=returns.index)
    market_features["Market_Avg_Return"] = returns.mean(axis=1)
    market_features["Market_Rolling_Vol"] = returns.mean(axis=1).rolling(
        rolling_window,
        min_periods=2
    ).std()
    market_features["Market_Momentum"] = returns.mean(axis=1).rolling(
        rolling_window,
        min_periods=2
    ).mean()

    parts = [returns, rolling_vol, rolling_momentum, market_features]

    if include_drawdown_features:
        parts.append(compute_drawdown_features(returns))

    if include_correlation_stress:
        parts.append(compute_correlation_stress_features(returns, rolling_window))

    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    aligned_returns = returns.loc[features.index].copy()

    return aligned_returns, features


def chronological_split(returns_df, features_df, train_ratio=0.70):
    n = len(features_df)
    split = int(n * train_ratio)

    if split < 5 or (n - split) < 3:
        print("WARNING: Very small train/test split. Results may be unstable.")

    returns_train = returns_df.iloc[:split].copy()
    returns_test = returns_df.iloc[split:].copy()

    features_train = features_df.iloc[:split].copy()
    features_test = features_df.iloc[split:].copy()

    return returns_train, returns_test, features_train, features_test


def build_observations(features_df, regime_df=None, use_regime=True):
    if use_regime and regime_df is not None:
        obs = pd.concat(
            [
                features_df.reset_index(drop=True),
                regime_df.reset_index(drop=True)
            ],
            axis=1
        )
    else:
        obs = features_df.reset_index(drop=True).copy()

    obs = obs.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return obs.astype(float)


def validate_alignment(name, returns_df, obs_df, regime_df=None):
    assert len(returns_df) == len(obs_df), (
        f"{name} misaligned: returns={len(returns_df)} obs={len(obs_df)}"
    )

    if regime_df is not None:
        assert len(returns_df) == len(regime_df), (
            f"{name} misaligned: returns={len(returns_df)} regime={len(regime_df)}"
        )


def scale_train_test_obs(obs_train, obs_test):
    scaler = StandardScaler()

    obs_train_scaled = pd.DataFrame(
        scaler.fit_transform(obs_train),
        columns=obs_train.columns
    )

    obs_test_scaled = pd.DataFrame(
        scaler.transform(obs_test),
        columns=obs_test.columns
    )

    return obs_train_scaled, obs_test_scaled, scaler
