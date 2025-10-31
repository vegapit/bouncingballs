import os, numpy, torch, gymnasium
from pathlib import Path
from game_environment import GameEnvironment
from expert_agent import ExpertAgent
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from custom_features_extractor import CustomFeaturesExtractor
import pickle

# Constants
DISPLAY_SHAPE = (480, 480)
FPS = 24
TOTAL_TIMESTEPS = 100_000
SEED = 13
NUM_EXPERT_EPISODES = 128

# Initialize environments
env = GameEnvironment(DISPLAY_SHAPE, 1.0 / float(FPS))
env.reset()

eval_env = GameEnvironment(DISPLAY_SHAPE, 1.0 / float(FPS))
eval_env.reset(seed=SEED)
eval_env = Monitor(eval_env)

# Paths
model_path = Path("ppo_bouncing_balls_latest.zip")

# Create or load model
if model_path.is_file():
    model = PPO.load(model_path, env=env)
    print("Loaded existing model.")
else:
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.999,
        clip_range=0.15,
        ent_coef=0.1
    )

# Set up logging
tmp_path = "./logs/"
os.makedirs(tmp_path, exist_ok=True)
new_logger = configure(tmp_path, ["stdout", "tensorboard"])
model.set_logger(new_logger)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

# Train with PPO (fine-tuning)
print("\nStarting PPO fine-tuning...")
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()
    eval_env.close()