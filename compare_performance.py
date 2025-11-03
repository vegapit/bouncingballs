import numpy
from tqdm import tqdm
from game_environment import GameEnvironment
from expert_agent import ExpertAgent
from stable_baselines3 import PPO

def action_selector(model, env, obs):
    if isinstance(model, ExpertAgent):
        action = model.select_action(env.hero_ball, env.balls)
    else:
        action, _ = model.predict(obs, deterministic=True)
    return action

def evaluate_model_performance(model, env, seed, n_episodes):
    """Evaluate a model over multiple episodes"""
    env.reset(seed)

    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm( range(n_episodes) ):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = action_selector(model, env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    return {
        'mean_reward': numpy.mean(episode_rewards),
        'std_reward': numpy.std(episode_rewards),
        'mean_length': numpy.mean(episode_lengths),
        'episodes': episode_rewards
    }

if __name__ == "__main__":
    DISPLAY_SHAPE = (480, 480)
    FPS = 24
    SEED = 17

    env = GameEnvironment(DISPLAY_SHAPE, 1.0 / float(FPS))
    
    model = PPO.load("ppo_bouncing_balls_latest", env=env)
    expert = ExpertAgent( env.motion_step, env.motion_step, env.dt )
    # Compare two models
    print("Evaluating DRL Model...")
    results1 = evaluate_model_performance(model, env, seed=SEED, n_episodes=100)

    print("Evaluating Expert Agent...")
    results2 = evaluate_model_performance(expert, env, seed=SEED, n_episodes=100)

    print(f"\nDRL model: {results1['mean_reward']:.3f} ± {results1['std_reward']:.3f}")
    print(f"Expert Agent: {results2['mean_reward']:.3f} ± {results2['std_reward']:.3f}")