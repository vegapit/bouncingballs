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

def pretrain_policy_masked(model, observations, actions, epochs=60, batch_size=256):
    """Pre-train with action 0 masked out during phase 1"""
    print(f"\n{'='*60}")
    print(f"Masked Progressive Pre-training")
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
    
    print(f"Phase 1: Training on minority classes (action 0 masked)")
    print(f"  Minority samples: {len(minority_obs)}")
    print(f"  Action 0 samples: {len(action0_obs)} (held out)\n")
    
    # Phase 1: Train with action 0 masked
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=5e-3)
    
    num_classes = model.action_space.n
    phase1_epochs = epochs // 2
    
    for epoch in range(phase1_epochs):
        indices = numpy.random.permutation(len(minority_obs))
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        action_correct = {i: 0 for i in range(1, num_classes)}
        action_total = {i: 0 for i in range(1, num_classes)}
        
        for start_idx in range(0, len(minority_obs), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_obs = {}
            for key in minority_obs[0].keys():
                obs_stack = numpy.stack([minority_obs[i][key] for i in batch_indices])
                batch_obs[key] = torch.FloatTensor(obs_stack).to(device)
            
            batch_actions = torch.LongTensor(
                [minority_actions[i] for i in batch_indices]
            ).to(device)
            
            # Get logits
            action_distribution = model.policy.get_distribution(batch_obs)
            logits = action_distribution.distribution.logits
            
            # CRITICAL: Mask action 0 by setting its logit to very negative value
            masked_logits = logits.clone()
            masked_logits[:, 0] = -1e10  # Effectively zero probability
            
            # Create new distribution with masked logits
            masked_probs = torch.softmax(masked_logits, dim=1)
            log_probs = torch.log(masked_probs + 1e-10)
            
            # Gather log probs for target actions
            selected_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = -selected_log_probs.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            
            # Calculate accuracy using masked predictions
            predicted_actions = masked_probs.argmax(dim=1)
            
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Phase 1 - Epoch {epoch+1:2d}/{phase1_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.3f}")
            
            if (epoch + 1) % 10 == 0:
                print("  Per-class accuracy:")
                for action in range(1, num_classes):
                    if action_total[action] > 0:
                        acc = action_correct[action] / action_total[action]
                        print(f"    Action {action}: {acc:.3f} "
                              f"({action_correct[action]}/{action_total[action]})")
    
    # Phase 2: Balanced training with all actions
    print(f"\nPhase 2: Training with balanced dataset (all actions)")
    
    # Undersample action 0
    num_minority = len(minority_obs)
    num_action0_to_keep = num_minority // 3
    
    indices_action0 = numpy.random.choice(len(action0_obs), num_action0_to_keep, replace=False)
    
    balanced_obs = minority_obs + [action0_obs[i] for i in indices_action0]
    balanced_actions = minority_actions + [action0_actions[i] for i in indices_action0]
    
    # Shuffle
    combined = list(zip(balanced_obs, balanced_actions))
    numpy.random.shuffle(combined)
    balanced_obs, balanced_actions = zip(*combined)
    balanced_obs = list(balanced_obs)
    balanced_actions = list(balanced_actions)
    
    print(f"  Total samples: {len(balanced_obs)}")
    action_counts = {}
    for action in balanced_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    print("  Action distribution:")
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        print(f"    Action {action}: {count:6d} ({100*count/len(balanced_actions):5.1f}%)")
    print()
    
    # Lower learning rate for phase 2
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    
    phase2_epochs = epochs - phase1_epochs
    
    for epoch in range(phase2_epochs):
        indices = numpy.random.permutation(len(balanced_obs))
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        action_correct = {i: 0 for i in range(num_classes)}
        action_total = {i: 0 for i in range(num_classes)}
        
        for start_idx in range(0, len(balanced_obs), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch_obs = {}
            for key in balanced_obs[0].keys():
                obs_stack = numpy.stack([balanced_obs[i][key] for i in batch_indices])
                batch_obs[key] = torch.FloatTensor(obs_stack).to(device)
            
            batch_actions = torch.LongTensor(
                [balanced_actions[i] for i in batch_indices]
            ).to(device)
            
            # Normal training (no masking)
            action_distribution = model.policy.get_distribution(batch_obs)
            log_probs = action_distribution.log_prob(batch_actions)
            loss = -log_probs.mean()
            
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
                per_class_accs.append(acc)
        balanced_accuracy = numpy.mean(per_class_accs)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Phase 2 - Epoch {epoch+1:2d}/{phase2_epochs} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.3f} | "
                  f"Balanced Acc: {balanced_accuracy:.3f}")
            
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
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            gamma=0.999,
            clip_range=0.15,
            ent_coef=0.1,
            device=device
        )
        
    # Evaluate before pre-training
    print("\nEvaluating random policy...")
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    print(f"Before pre-training: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    
    try:
        # Pre-train with expert data
        pretrain_policy_masked(model, observations, actions, epochs=100, batch_size=256)
    except Exception as e:
        print(f"BC Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Evaluate after pre-training
    print("\nEvaluating pre-trained policy...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    print(f"After pre-training: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")
    print(f"Improvement: {mean_reward_after - mean_reward_before:.2f}")