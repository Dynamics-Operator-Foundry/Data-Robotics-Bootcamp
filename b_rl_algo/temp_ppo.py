import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# Hyperparameters
gamma = 0.99
lambda_gae = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
learning_rate = 3e-4
epochs = 10
batch_size = 64
rollout_steps = 2048

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean_layer(x)
        std = self.log_std.exp()
        return mean, std

# Define the Value Function Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)

# PPO Algorithm
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_function = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=learning_rate)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, log_probs, rewards, values, dones):
        advantages = self.compute_advantages(rewards, values, dones)
        returns = torch.tensor(advantages) + torch.tensor(values[:-1])
        advantages = torch.tensor(advantages).detach()

        for _ in range(epochs):
            for i in range(0, len(states), batch_size):
                state_batch = torch.tensor(states[i:i+batch_size], dtype=torch.float32)
                action_batch = torch.tensor(actions[i:i+batch_size], dtype=torch.float32)
                old_log_probs_batch = torch.tensor(log_probs[i:i+batch_size], dtype=torch.float32)
                advantage_batch = advantages[i:i+batch_size]
                return_batch = returns[i:i+batch_size]

                # Compute new policy log_probs
                mean, std = self.policy(state_batch)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(action_batch).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)

                # Compute policy loss
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()

                # Compute value loss
                value_loss = value_coef * (self.value_function(state_batch).squeeze() - return_batch).pow(2).mean()

                # Optimize policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Optimize value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob.detach().item()

# Main Training Loop
def train(env, ppo_agent, num_episodes):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    for episode in range(num_episodes):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        state = env.reset()
        done = False

        while not done:
            action, log_prob = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(ppo_agent.value_function(torch.tensor(state, dtype=torch.float32)).item())

            state = next_state

        # Add value of the final state for bootstrapping
        values.append(ppo_agent.value_function(torch.tensor(state, dtype=torch.float32)).item())

        # Update PPO
        ppo_agent.update(states, actions, log_probs, rewards, values, dones)

        print(f"Episode {episode + 1}/{num_episodes} completed.")

# Example Usage
if __name__ == "__main__":
    import gym

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = PPO(state_dim, action_dim)
    train(env, ppo_agent, num_episodes=1000)
