from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils import set_seed


def train_ppo(env_fn, seed=42, total_timesteps=50_000):
    set_seed(seed)

    vec_env = DummyVecEnv([env_fn])
    vec_env.seed(seed)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        verbose=0,
        learning_rate=1e-4,
        gamma=0.90,
        gae_lambda=0.92,
        n_steps=32,
        batch_size=32,
        ent_coef=0.01,
        clip_range=0.20,
        n_epochs=30
    )

    model.learn(total_timesteps=total_timesteps)

    return model
