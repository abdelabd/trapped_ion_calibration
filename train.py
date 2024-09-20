import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Locals
from env import TrappedIonEnv
from agent import PPOAgent

DEBUG = False

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_episode_intensity(episode_intensities, target_intensity, filename, episode=None):
    fig = plt.figure()
    plt.plot(range(len(episode_intensities)), episode_intensities)
    plt.plot(range(len(episode_intensities)), [target_intensity]*len(episode_intensities), linestyle='--', label = "Target Intensity")
    plt.xlabel('Step')
    plt.ylabel('Laser Intensity (W/m^2)')
    plt.title(f"Final training episode = {episode+1}")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def save_n_episode_intensity(n, all_episode_intensities, target_intensity, filename, episode=None):
    fig = plt.figure()
    for i in range(n-1, 0, -1):
        episode_intensities = all_episode_intensities[i]
        plt.plot(range(len(episode_intensities)), episode_intensities, color="r", alpha = 0.1)
    plt.plot(range(len(episode_intensities)), [target_intensity]*len(episode_intensities), linestyle='--', label = "Target Intensity")
    plt.xlabel('Step')
    plt.ylabel('Laser Intensity (W/m^2)')
    plt.title(f"Final training episode = {episode+1}")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def lr_schedule(episode):
    initial_lr = 1e-4
    min_lr = 1e-7
    decay_factor = 0.995
    return max(initial_lr * (decay_factor ** episode), min_lr)

# Training loop
def train_ppo(env, agent, num_episodes, max_steps):
    all_rewards = []
    all_losses = []
    all_intensities = []
    all_lrs = []

    try:

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
            episode_intensities = []

            for step in range(max_steps):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                episode_intensities.append(env.laser_intensity)

                state = next_state
                episode_reward += reward

                if done:
                    break
            
            all_rewards.append(episode_reward)
            loss = agent.update(states, actions, log_prob, rewards, next_states, dones)
            all_losses.append(loss)
            all_intensities.append(episode_intensities)

            # Update learning rate
            agent.update_learning_rate(episode)
            all_lrs.append(agent.current_lr)

            if (episode % 10 == 0)|(episode==num_episodes-1):
                avg_reward = np.mean(all_rewards[-100:])
                std_reward = np.std(all_rewards[-100:])
                avg_loss = np.mean(all_losses[-100:])
                std_loss = np.std(all_losses[-100:])
                avg_intensity = np.mean([intensities[-1] for intensities in all_intensities[-100:]])
                std_intensity = np.std([intensities[-1] for intensities in all_intensities[-100:]])
                print(f"\nEpisode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Avg Loss (last 100): {avg_loss:.4f}, Avg Final Intensity (last 100): {avg_intensity:.2f}, Std final intensity (last 100): {std_intensity:.2f}")
                print(f"Last action: {action}, Last reward: {reward:.2f}, Initial intensity: {env.initial_intensity:.2f}, Final intensity: {env.laser_intensity:.2f}")

            if episode==num_episodes-1:
                save_n_episode_intensity(100, all_intensities, env.target_intensity, "figures/final_100_training_episode.png", episode)
                save_episode_intensity(episode_intensities, env.target_intensity, "figures/final_training_episode.png", episode)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        avg_reward = np.mean(all_rewards[-100:])
        avg_loss = np.mean(all_losses[-100:])
        avg_intensity = np.mean([intensities[-1] for intensities in all_intensities[-100:]])
        std_intensity = np.std([intensities[-1] for intensities in all_intensities[-100:]])
        print(f"Avg Reward (last 100): {avg_reward:.2f}, Avg Loss (last 100): {avg_loss:.4f}, Avg Final Intensity (last 100): {avg_intensity:.2f}, Std final intensity (last 100): {std_intensity:.2f}")
        print(f"Last reward: {all_rewards[-1]:.2f}, Final intensity: {env.laser_intensity:.2f}")
        save_n_episode_intensity(100, all_intensities, env.target_intensity, "figures/final_100_training_episode.png", episode)
        save_episode_intensity(episode_intensities, env.target_intensity, "figures/final_training_episode.png", episode)
        
            

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 777
    set_random_seed(seed)
    env = TrappedIonEnv(seed=seed)

    input_dim = env.observation_space.shape[0]
    hidden_dim = 64
    output_dim = env.action_space.shape[0]

    agent = PPOAgent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        initial_lr=1e-4,
        gamma=0.99,
        clip_epsilon=0.3,
        epochs=10,
        lr_schedule = lr_schedule
    )

    if DEBUG:
        num_episodes = 100
    else:
        num_episodes = int(1e3)
    max_steps = 100

    train_ppo(env, agent, num_episodes, max_steps)
    torch.save(agent.ac_model.state_dict(), "models/ppo_model.pth")
    print("Model saved!")
