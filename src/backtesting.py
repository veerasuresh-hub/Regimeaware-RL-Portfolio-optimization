import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.utils import safe_softmax


def backtest_rl(model, returns_df, obs_df, transaction_cost=0.001):
    value = 1.0
    values = [value]
    period_returns = []
    weights_history = []
    turnover_history = []

    prev_weights = None

    for i in range(len(returns_df) - 1):
        obs = obs_df.iloc[i].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        weights = safe_softmax(action)

        market_return = np.append(returns_df.iloc[i + 1].values, 0.0)
        ret = float(np.dot(weights, market_return))

        if prev_weights is None:
            turnover = 0.0
        else:
            turnover = float(np.sum(np.abs(weights - prev_weights)))
            ret -= transaction_cost * turnover

        value *= (1.0 + ret)

        values.append(value)
        period_returns.append(ret)
        weights_history.append(weights)
        turnover_history.append(turnover)

        prev_weights = weights

    weights_df = pd.DataFrame(
        weights_history,
        columns=list(returns_df.columns) + ["Cash"]
    )

    return {
        "values": np.array(values),
        "returns": np.array(period_returns),
        "weights": weights_df,
        "turnover": np.array(turnover_history)
    }


def backtest_equal_weight(returns_df):
    n = returns_df.shape[1]
    weights = np.ones(n) / n

    value = 1.0
    values = [value]
    period_returns = []

    for i in range(len(returns_df) - 1):
        ret = float(np.dot(weights, returns_df.iloc[i + 1].values))

        value *= (1.0 + ret)

        values.append(value)
        period_returns.append(ret)

    return {
        "values": np.array(values),
        "returns": np.array(period_returns)
    }


def backtest_markowitz(returns_df, window=5):
    n = returns_df.shape[1]

    value = 1.0
    values = [value]
    period_returns = []
    weights_history = []

    for i in range(len(returns_df) - 1):
        if i < max(2, window):
            weights = np.ones(n) / n
        else:
            window_data = returns_df.iloc[i - window:i]
            mu = window_data.mean().values
            cov = np.cov(window_data.T)
            cov = np.atleast_2d(cov)

            try:
                raw = np.linalg.solve(cov + 1e-4 * np.eye(n), mu)
            except Exception:
                raw = np.ones(n)

            raw = np.clip(raw, 0, None)
            weights = np.ones(n) / n if raw.sum() <= 1e-12 else raw / raw.sum()

        ret = float(np.dot(weights, returns_df.iloc[i + 1].values))

        value *= (1.0 + ret)

        values.append(value)
        period_returns.append(ret)
        weights_history.append(weights)

    return {
        "values": np.array(values),
        "returns": np.array(period_returns),
        "weights": pd.DataFrame(weights_history, columns=returns_df.columns)
    }


def get_quasi_diag(link):
    link = link.astype(int)

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)

        df0 = sort_ix[sort_ix >= num_items]

        i = df0.index
        j = df0.values - num_items

        sort_ix[i] = link[j, 0]

        df1 = pd.Series(link[j, 1], index=i + 1)

        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def get_cluster_var(cov, cluster_items):
    cov_slice = cov.loc[cluster_items, cluster_items]

    inv_diag = 1.0 / np.diag(cov_slice)
    weights = inv_diag / inv_diag.sum()

    cluster_var = np.dot(weights, np.dot(cov_slice, weights))

    return cluster_var


def recursive_bisection(cov, sorted_items):
    weights = pd.Series(1.0, index=sorted_items)
    clusters = [sorted_items]

    while len(clusters) > 0:
        clusters = [
            cluster[start:end]
            for cluster in clusters
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]

        for i in range(0, len(clusters), 2):
            cluster_1 = clusters[i]
            cluster_2 = clusters[i + 1]

            var_1 = get_cluster_var(cov, cluster_1)
            var_2 = get_cluster_var(cov, cluster_2)

            alpha = 1.0 - var_1 / (var_1 + var_2)

            weights[cluster_1] *= alpha
            weights[cluster_2] *= 1.0 - alpha

    return weights


def hrp_weights(returns_window):
    cov = returns_window.cov()
    corr = returns_window.corr()

    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = corr.clip(-1, 1)

    distance = np.sqrt((1 - corr) / 2)
    distance = distance.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    condensed_distance = squareform(distance.values, checks=False)
    link = linkage(condensed_distance, method="single")

    sorted_index = get_quasi_diag(link)
    sorted_items = corr.index[sorted_index].tolist()

    weights = recursive_bisection(cov, sorted_items)
    weights = weights.reindex(returns_window.columns).fillna(0.0)

    weights = weights.clip(lower=0)
    weights = weights / weights.sum()

    return weights.values


def backtest_hrp(returns_df, window=5):
    n = returns_df.shape[1]

    value = 1.0
    values = [value]
    period_returns = []
    weights_history = []

    for i in range(len(returns_df) - 1):
        if i < max(3, window):
            weights = np.ones(n) / n
        else:
            returns_window = returns_df.iloc[i - window:i]

            try:
                weights = hrp_weights(returns_window)
            except Exception:
                weights = np.ones(n) / n

        ret = float(np.dot(weights, returns_df.iloc[i + 1].values))

        value *= (1.0 + ret)

        values.append(value)
        period_returns.append(ret)
        weights_history.append(weights)

    return {
        "values": np.array(values),
        "returns": np.array(period_returns),
        "weights": pd.DataFrame(weights_history, columns=returns_df.columns)
    }


def backtest_risk_parity(returns_df, window=5):
    n = returns_df.shape[1]

    value = 1.0
    values = [value]
    period_returns = []
    weights_history = []

    for i in range(len(returns_df) - 1):
        if i < max(2, window):
            weights = np.ones(n) / n
        else:
            window_data = returns_df.iloc[i - window:i]
            vol = window_data.std().values
            inv_vol = 1.0 / (vol + 1e-8)
            weights = inv_vol / inv_vol.sum()

        ret = float(np.dot(weights, returns_df.iloc[i + 1].values))

        value *= (1.0 + ret)

        values.append(value)
        period_returns.append(ret)
        weights_history.append(weights)

    return {
        "values": np.array(values),
        "returns": np.array(period_returns),
        "weights": pd.DataFrame(weights_history, columns=returns_df.columns)
    }
