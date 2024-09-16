import numpy as np
from env import TrappedIonEnv
from agent import PPOAgent

# Training loop
def train_ppo(env, agent, num_episodes, max_steps):
    all_rewards = []
    all_losses = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        loss = agent.update(states, actions, rewards, next_states, dones)

        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_loss = np.mean(all_losses[-100:])
            print(f"\nEpisode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Avg Loss (last 100): {avg_loss:.4f}")
            print(f"Last action: {action}, Last reward: {reward:.2f}, Final laser intensity: {env.laser_intensity:.2f}")

if __name__ == "__main__":
    env = TrappedIonEnv()
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

    num_episodes = 1000
    max_steps = 100

    train_ppo(env, agent, num_episodes, max_steps)