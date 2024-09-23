import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Beta
import numpy as np

class MLPActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2), # Output mu and std
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        mu_std = self.actor(state)
        mu, std = mu_std.chunk(2, dim=-1)
        std = F.softplus(std) + 1e-5  # Ensure std > 0
        return mu, std, value

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, initial_lr, gamma, clip_epsilon, epochs, lr_schedule):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"self.device: {self.device}")

        self.ac_model = MLPActorCritic(input_dim, hidden_dim, output_dim).to(self.device)

        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=initial_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.exploration_noise = 0.1  # Add exploration noise
        self.lr_schedule = lr_schedule
        self.current_lr = initial_lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, std, _ = self.ac_model(state)
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):

        # Skip update if there's insufficient data
        if len(rewards) < 2:
            return 0.0  # Return a default loss value or None
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Normalize rewards safely
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        if rewards_std > 1e-8:
            rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)
        else:
            rewards = rewards - rewards_mean  # Zero-mean normalization


        # Compute advantages
        with torch.no_grad():
            _, _, next_values = self.ac_model(next_states)
            _, _, values = self.ac_model(states)
            delta = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = delta.detach()
        
        # Normalize advantages safely
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std > 1e-8:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean  # Zero-mean normalization


        # PPO update
        for _ in range(self.epochs):
            mu, std, values = self.ac_model(states)

            # Add numerical stability
            dist = Normal(mu, std)

            log_probs = dist.log_prob(actions).sum(1, keepdim=True)
            entropy = dist.entropy().mean()

            ratio = (log_probs - old_log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Decay exploration noise
        self.exploration_noise *= 0.995

        return loss.item()