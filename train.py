import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Locals
from env import TrappedIonEnv
from agent import PPOAgent

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training loop
def train_ppo(env, agent, num_episodes, max_steps):
    all_rewards = []
    all_losses = []
    all_intensities = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_intensities = []

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            episode_intensities.append(env.laser_intensity)

            state = next_state
            episode_reward += reward

            if done:
                break
        
        all_rewards.append(episode_reward)
        loss = agent.update(states, actions, rewards, next_states, dones)
        all_losses.append(loss)
        all_intensities.append(episode_intensities)

        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_loss = np.mean(all_losses[-100:])
            avg_intensity = np.mean([intensities[-1] for intensities in all_intensities[-100:]])
            print(f"\nEpisode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Avg Loss (last 100): {avg_loss:.4f}, Avg Final Intensity (last 100): {avg_intensity:.2f}")
            print(f"Last action: {action}, Last reward: {reward:.2f}, Final intensity: {env.laser_intensity:.2f}")

        if episode==num_episodes-1:
            fig = plt.figure()
            plt.plot(range(len(episode_intensities)), episode_intensities)
            plt.xlabel('Step')
            plt.ylabel('Laser Intensity')
            plt.title("Final training episode")
            plt.savefig("figures/final_training_episode.png")
            plt.close(fig)

            print(f"Last action: {action}, Last reward: {reward:.2f}, Final intensity: {env.laser_intensity:.2f}")

if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 777
    set_random_seed(seed)
    env = TrappedIonEnv()
    env.seed(seed)  # Set seed for the environment

    input_dim = env.observation_space.shape[0]
    hidden_dim = 64
    output_dim = env.action_space.shape[0]

    agent = PPOAgent(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10
    )

    num_episodes = 700
    max_steps = 100

    train_ppo(env, agent, num_episodes, max_steps)
    torch.save(agent.ac_model.state_dict(), "models/ppo_model.pth")
    print("Model saved!")
    