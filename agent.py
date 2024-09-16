import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class MLPActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        return mu, std, value

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, gamma, clip_epsilon, epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"self.device: {self.device}")
        self.ac_model = MLPActorCritic(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, std, _ = self.ac_model(state)
        dist = Normal(mu, std)
        action = dist.sample()
        return action.cpu().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Compute advantages
        with torch.no_grad():
            _, _, next_values = self.ac_model(next_states)
            _, _, values = self.ac_model(states)
            delta = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = delta.detach()

        # PPO update
        for _ in range(self.epochs):
            mu, std, values = self.ac_model(states)
            dist = Normal(mu, std)
            
            log_probs = dist.log_prob(actions).sum(1, keepdim=True)
            entropy = dist.entropy().mean()

            ratio = (log_probs - log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
            self.optimizer.step()

        return loss.item()