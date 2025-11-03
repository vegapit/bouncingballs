import os
from pathlib import Path
from game_environment import GameEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from custom_features_extractor import CustomFeaturesExtractor

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
        learning_rate=1e-5,
        n_steps=512,
        batch_size=128,
        gamma=0.999,
        clip_range=0.05,
        ent_coef=0.01
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
    n_eval_episodes=15
)

# Train with PPO (fine-tuning)
print("\nStarting PPO fine-tuning...")
try:
    # Phase 1: Train only value function
    print("Phase 1: Train value function only...")
    for param in model.policy.action_net.parameters():
        param.requires_grad = False
    for param in model.policy.mlp_extractor.policy_net.parameters():
        param.requires_grad = False

    model.vf_coef = 1.0
    model.learn(total_timesteps=TOTAL_TIMESTEPS // 5)

    # Phase 2: Unfreeze and train everything
    print("Phase 2: Training full policy...")
    for param in model.policy.parameters():
        param.requires_grad = True
        
    model.vf_coef = 0.5
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