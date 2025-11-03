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

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def test_expert_performance(env, expert_agent, num_episodes=10):
    """Verify expert episode returns"""
    episode_returns = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False
        step_count = 0
        
        while not done and step_count < 5000:  # Safety limit
            action = expert_agent.select_action(env.hero_ball, env.balls)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
            step_count += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        print(f"Episode {episode+1}: Return = {episode_return:.2f}, Length = {step_count}")
    
    print(f"\nExpert Performance:")
    print(f"  Mean return: {numpy.mean(episode_returns):.2f} +/- {numpy.std(episode_returns):.2f}")
    print(f"  Mean length: {numpy.mean(episode_lengths):.1f}")
    print(f"  Min/Max return: [{numpy.min(episode_returns):.2f}, {numpy.max(episode_returns):.2f}]")
    
    return numpy.mean(episode_returns)

def collect_expert_data(env, expert_agent, num_episodes=100):
    """Collect expert demonstrations with episode tracking"""
    observations = []
    actions = []
    episode_returns = []
    episode_lengths = []
    
    action_counts = {}
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_done = False
        episode_return = 0
        episode_length = 0
        
        while not episode_done:
            action = expert_agent.select_action(env.hero_ball, env.balls)
            
            action_counts[action] = action_counts.get(action, 0) + 1
            observations.append(obs)
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            episode_done = terminated or truncated
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode+1}/{num_episodes} episodes | "
                  f"Last return: {episode_return:.2f} | "
                  f"Mean return: {numpy.mean(episode_returns):.2f}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Expert Data Collection Complete")
    print(f"{'='*60}")
    print(f"Episodes: {len(episode_returns)}")
    print(f"Total transitions: {len(observations)}")
    print(f"Mean episode return: {numpy.mean(episode_returns):.2f} +/- {numpy.std(episode_returns):.2f}")
    print(f"Mean episode length: {numpy.mean(episode_lengths):.1f}")
    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        print(f"  Action {action}: {count:6d} ({100*count/total_actions:5.1f}%)")
    print(f"{'='*60}\n")
    
    return observations, actions, episode_returns, episode_lengths

def pretrain_policy_balanced(model, observations, actions, max_epochs, batch_size, min_accuracy):
    print(f"\n{'='*60}")
    print(f"Behavioural Cloning / Pre-training")
    print(f"{'='*60}\n")
    
    device = model.device
    
    # Reinitialize action head
    print("Reinitializing action network...")
    for layer in model.policy.action_net.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)
    
    model.policy.train()
    
    # Split data
    minority_obs = []
    minority_actions = []
    action0_obs = []
    action0_actions = []
    
    for i, action in enumerate(actions):
        if action == 0:
            action0_obs.append(observations[i])
            action0_actions.append(action)
        else:
            minority_obs.append(observations[i])
            minority_actions.append(action)
    
    print(f"Minority samples: {len(minority_obs)}")
    print(f"Action 0 samples: {len(action0_obs)}\n")
    
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    num_classes = model.action_space.n
    
    # Shuffle
    combined_obs = action0_obs + minority_obs
    combined_actions = action0_actions + minority_actions
    combined = list(zip(combined_obs, combined_actions))
    
    total_samples = len(combined)

    print(f"Total samples: {total_samples}")
    action_counts = {}
    for action in combined_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    action_weights = torch.zeros( len(action_counts.keys()), dtype=torch.float32 ).to(device)
    
    print("Action distribution:")
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        weight = float(total_samples) / float(count)
        action_weights[action] = weight
        print(f"  Action {action}: {count:6d} ({100/weight:5.1f}%)")
    print()

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    for epoch in range(max_epochs):
        indices = numpy.random.permutation(len(combined_obs))
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        action_correct = {i: 0 for i in range(num_classes)}
        action_total = {i: 0 for i in range(num_classes)}
        
        for start_idx in range(0, len(combined_obs), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_obs = {}
            for key in combined_obs[0].keys():
                obs_stack = numpy.stack([combined_obs[i][key] for i in batch_indices])
                batch_obs[key] = torch.FloatTensor(obs_stack).to(device)
            
            batch_actions = torch.LongTensor(
                [combined_actions[i] for i in batch_indices]
            ).to(device)
            
            # Frequency adjusted loss calculation
            action_distribution = model.policy.get_distribution(batch_obs)
            log_probs = action_distribution.log_prob(batch_actions)
            weights = action_weights.gather(0, batch_actions)
            loss = -torch.mean( weights * log_probs )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            
            predicted_actions = action_distribution.distribution.probs.argmax(dim=1)
            
            for i in range(len(batch_actions)):
                true_action = batch_actions[i].item()
                pred_action = predicted_actions[i].item()
                action_total[true_action] += 1
                if true_action == pred_action:
                    action_correct[true_action] += 1
            
            accuracy = (predicted_actions == batch_actions).float().mean().item()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        per_class_accs = []
        for action in range(num_classes):
            if action_total[action] > 0:
                acc = action_correct[action] / action_total[action]
                per_class_accs.append( acc )
        balanced_accuracy = numpy.mean(per_class_accs)

        print(f"Epoch {epoch+1:2d}/{max_epochs} | "
            f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.3f} | "
            f"Balanced Acc: {balanced_accuracy:.3f}")

        if balanced_accuracy > min_accuracy:
            break

        if (epoch + 1) % 10 == 0:
            print("  Per-class accuracy:")
            for action in range(num_classes):
                if action_total[action] > 0:
                    acc = action_correct[action] / action_total[action]
                    print(f"    Action {action}: {acc:.3f} "
                            f"({action_correct[action]}/{action_total[action]})")
            
    
    print(f"\n{'='*60}")
    print(f"Pre-training complete!")
    print(f"{'='*60}\n")
    
    model.policy.eval()

if __name__ == "__main__":

    device = get_device()

    # Initialize environments
    env = GameEnvironment(DISPLAY_SHAPE, 1.0 / float(FPS))
    env.reset()

    eval_env = GameEnvironment(DISPLAY_SHAPE, 1.0 / float(FPS))
    eval_env.reset(seed=SEED)
    eval_env = Monitor(eval_env)

    # Paths
    model_path = Path("ppo_bouncing_balls_latest.zip")
    expert_data_path = Path("expert_demonstrations.pkl")

    # Collect or load expert demonstrations
    if expert_data_path.exists():
        print("Loading existing expert demonstrations...")
        with open(expert_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        observations = expert_data['observations']
        actions = expert_data['actions']
    else:
        print("Collecting expert demonstrations...")
        expert_agent = ExpertAgent( env.motion_step, env.motion_step, env.dt )
        expert_mean_return = test_expert_performance(env, expert_agent, num_episodes=10)
        observations, actions, rewards, dones = collect_expert_data(
            env, expert_agent, NUM_EXPERT_EPISODES
        )
        
        # Save for future use
        with open(expert_data_path, 'wb') as f:
            pickle.dump({
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'dones': dones
            }, f)
        print(f"Saved {len(observations)} expert transitions")
        
        # Print expert statistics
        print(f"\nExpert Statistics:")
        print(f"  Total transitions: {len(observations)}")
        print(f"  Average reward: {numpy.mean(rewards):.2f}")
        print(f"  Episodes: {sum(dones)}")

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
            n_steps=2048,
            batch_size=128,
            gamma=0.999,
            clip_range=0.05,
            ent_coef=0.0001,
            vf_coef=2.0,
            max_grad_norm=0.5,
            n_epochs=17,
            device=device
        )
        
    try:
        # Pre-train with expert data
        pretrain_policy_balanced(model, observations, actions, max_epochs=200, batch_size=256, min_accuracy=0.95)
    except Exception as e:
        print(f"BC Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.save(model_path)
        print(f"Model saved to {model_path}")