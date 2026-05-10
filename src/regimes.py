import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


def fit_hmm_regimes(
    features_train,
    features_test,
    returns_train,
    n_regimes=3,
    hmm_iter=500,
    seed=42
):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(features_train)
    X_test = scaler.transform(features_test)

    effective_regimes = min(
        n_regimes,
        max(2, len(features_train) // 3)
    )

    if effective_regimes != n_regimes:
        print(
            f"WARNING: Reduced HMM regimes from "
            f"{n_regimes} to {effective_regimes} due to small sample size."
        )

    hmm = GaussianHMM(
        n_components=effective_regimes,
        covariance_type="diag",
        n_iter=hmm_iter,
        random_state=seed
    )

    hmm.fit(X_train)

    train_states = hmm.predict(X_train)

    avg_asset_return = returns_train.mean(axis=1)

    feature_cols = features_train.columns

    vol_cols = [c for c in feature_cols if "Vol" in c]
    dd_cols = [c for c in feature_cols if "DD" in c]
    corr_cols = [c for c in feature_cols if "Corr" in c]
    mom_cols = [c for c in feature_cols if "Mom" in c]

    state_scores = []

    for state in range(effective_regimes):
        mask = train_states == state

        if mask.sum() == 0:
            state_scores.append(-999)
            continue

        state_features = features_train.iloc[mask]
        state_returns = avg_asset_return.iloc[mask]

        avg_return = state_returns.mean()

        avg_vol = (
            state_features[vol_cols].mean().mean()
            if len(vol_cols) > 0
            else 0.0
        )

        avg_dd = (
            state_features[dd_cols].mean().mean()
            if len(dd_cols) > 0
            else 0.0
        )

        avg_corr = (
            state_features[corr_cols].mean().mean()
            if len(corr_cols) > 0
            else 0.0
        )

        avg_mom = (
            state_features[mom_cols].mean().mean()
            if len(mom_cols) > 0
            else 0.0
        )

        stress_score = (
            -avg_return
            + avg_vol
            + abs(avg_dd)
            + avg_corr
            - avg_mom
        )

        state_scores.append(stress_score)

    order = np.argsort(state_scores)

    if effective_regimes == 2:
        label_map = {
            order[0]: "Bull",
            order[1]: "Bear"
        }
        ordered_cols = ["Bull", "Bear", "Crisis"]

    elif effective_regimes >= 3:
        label_map = {
            order[0]: "Bull",
            order[1]: "Bear",
            order[2]: "Crisis"
        }
        ordered_cols = ["Bull", "Bear", "Crisis"]

    else:
        raise ValueError("HMM requires at least 2 regimes.")

    print("\nHMM state stress scores:")
    for state, score in enumerate(state_scores):
        print(
            f"State {state}: Stress Score = {score:.4f}, "
            f"Label = {label_map[state]}"
        )

    prob_train_raw = hmm.predict_proba(X_train)
    prob_test_raw = hmm.predict_proba(X_test)

    regime_train = pd.DataFrame(index=features_train.index)
    regime_test = pd.DataFrame(index=features_test.index)

    for state in range(effective_regimes):
        col = label_map[state]
        regime_train[col] = prob_train_raw[:, state]
        regime_test[col] = prob_test_raw[:, state]

    regime_train = regime_train.reindex(
        columns=ordered_cols,
        fill_value=0.0
    )

    regime_test = regime_test.reindex(
        columns=ordered_cols,
        fill_value=0.0
    )

    return hmm, scaler, regime_train, regime_test
