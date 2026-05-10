import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils import safe_softmax


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_df,
        obs_df,
        regime_df=None,
        transaction_cost=0.0,
        volatility_penalty=0.0,
        crisis_penalty=0.0,
        reward_clip=None,
    ):
        super().__init__()

        self.returns = returns_df.reset_index(drop=True).astype(float)
        self.obs_data = obs_df.reset_index(drop=True).astype(float)
        self.regime = None if regime_df is None else regime_df.reset_index(drop=True).astype(float)

        self.transaction_cost = transaction_cost
        self.volatility_penalty = volatility_penalty
        self.crisis_penalty = crisis_penalty
        self.reward_clip = reward_clip

        self.n_risky_assets = self.returns.shape[1]
        self.n_assets = self.n_risky_assets + 1

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_data.shape[1],),
            dtype=np.float32
        )

        self.current_step = 0
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_return_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_return_history = []

        return self._get_obs(), {}

    def step(self, action):
        weights = safe_softmax(action)
        next_step = self.current_step + 1

        if next_step >= len(self.returns):
            return self._get_obs(), 0.0, True, False, {}

        market_return = np.append(
            self.returns.iloc[next_step].values,
            0.0
        )

        gross_return = float(np.dot(weights, market_return))

        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        cost = self.transaction_cost * turnover

        net_return = gross_return - cost
        self.portfolio_return_history.append(net_return)

        if len(self.portfolio_return_history) > 1:
            recent = self.portfolio_return_history[-3:]
            realised_vol = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
        else:
            realised_vol = 0.0

        turnover_penalty = 0.01 * turnover

        reward = net_return
        reward -= self.volatility_penalty * realised_vol
        reward -= turnover_penalty

        if self.regime is not None and "Crisis" in self.regime.columns:
            crisis_prob = float(self.regime.iloc[self.current_step]["Crisis"])
            downside_loss = max(-net_return, 0.0)

            reward -= self.crisis_penalty * crisis_prob * downside_loss
            reward -= 0.01 * max(weights[-1] - 0.4, 0)

        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.prev_weights = weights
        self.current_step = next_step

        terminated = self.current_step >= len(self.returns) - 1
        truncated = False

        info = {
            "net_return": net_return,
            "turnover": turnover,
            "weights": weights
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        return self.obs_data.iloc[self.current_step].values.astype(np.float32)


def make_env(
    returns_df,
    obs_df,
    regime_df=None,
    transaction_cost=0.0,
    volatility_penalty=0.0,
    crisis_penalty=0.0,
    reward_clip=None
):
    return lambda: PortfolioEnv(
        returns_df=returns_df,
        obs_df=obs_df,
        regime_df=regime_df,
        transaction_cost=transaction_cost,
        volatility_penalty=volatility_penalty,
        crisis_penalty=crisis_penalty,
        reward_clip=reward_clip,
    )
