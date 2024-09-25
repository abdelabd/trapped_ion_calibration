import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(hidden_dim, output_dim * 2),  # Output mu and std
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
    def __init__(self, input_dim, hidden_dim, output_dim, initial_lr, gamma, clip_epsilon, lam, epochs, lr_schedule):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"self.device: {self.device}")

        self.ac_model = MLPActorCritic(input_dim, hidden_dim, output_dim).to(self.device)

        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=initial_lr)
        self.gamma = gamma
        self.lam = lam # 0.95  # GAE lambda parameter
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.lr_schedule = lr_schedule
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, std, _ = self.ac_model(state)
        dist = Normal(mu, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def compute_gae(self, rewards, masks, values, next_value):
        # Reshape tensors to ensure they are 1-dimensional
        values = values.view(-1)
        next_value = next_value.view(-1)
        
        # Concatenate along the correct dimension
        values = torch.cat([values, next_value], dim=0)
        
        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            gae = delta + self.gamma * self.lam * masks[t] * gae
            advantages[t] = gae
        return advantages


    def update(self, states, actions, old_log_probs, rewards, next_states, dones):

        # Skip update if there's insufficient data
        if len(rewards) < 2:
            return 0.0  # Return a default loss value or None

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Compute masks
        masks = 1 - dones

        # Compute values and next_values
        with torch.no_grad():
            _, _, values = self.ac_model(states)
            _, _, next_values = self.ac_model(next_states)

        values = values.squeeze(-1)
        next_values = next_values.squeeze(-1)

        if dones[-1]:
            next_value = torch.zeros(1).to(self.device)
        else:
            next_value = next_values[-1]

        # Compute advantages using GAE
        advantages = self.compute_gae(rewards, masks, values, next_value)

        # Compute returns
        returns = advantages + values

        # Normalize advantages safely
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std > 1e-8:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean  # Zero-mean normalization

        # PPO update
        for _ in range(self.epochs):
            mu, std, values_pred = self.ac_model(states)
            values_pred = values_pred.squeeze(-1)

            dist = Normal(mu, std)

            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values_pred).pow(2).mean()
            loss = actor_loss + critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Decay exploration noise (if applicable)
        # self.exploration_noise *= 0.995

        return loss.item()
