# Initialize the environment and the agent
import gym
from collections import deque
import random

# Set up the environment
env = gym.make("CartPole-v1")

# Define training parameters
num_episodes = 250
max_steps_per_episode = 200
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_rate = 0.99
gamma = 0.9
lr = 0.0025
buffer_size = 10000
buffer = deque(maxlen=buffer_size)
batch_size = 128
update_frequency = 10


# Initialize the DQNAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
new_agent = DQNAgent(input_dim, output_dim, seed=170715, lr = lr)


# Training loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

    # Run one episode
    for step in range(max_steps_per_episode):
        # Choose and perform an action
        action = new_agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        
        buffer.append((state, action, reward, next_state, done))
        
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            # Update the agent's knowledge
            new_agent.learn(batch, gamma)
        
        state = next_state
        
        # Check if the episode has ended
        if done:
            break
    
    if (episode + 1) % update_frequency == 0:
        print(f"Episode {episode + 1}: Finished training")