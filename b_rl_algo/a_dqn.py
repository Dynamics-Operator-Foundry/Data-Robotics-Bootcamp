import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as funcs
import random

from rl_lib import ReplayBuffer as buffer, QNN as qnn


class DQN:
    def __init__(
        self,
        n_state,
        n_action,
        seed,
        lr,
        device="cpu"
    ):
        self.n_state = n_state
        self.n_action = n_action
        self.seed = random.seed(seed)
        self.device = device
        
        self.qnn_hat = qnn(
            n_state=n_state,
            n_action=n_action,
            seed=seed
        ).to(device)
        
        self.qnn_target = qnn(
            n_state=n_state,
            n_action=n_action,
            seed=seed
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.qnn_hat.parameters(), 
            lr
        )
        
        self.qnn_deploy = qnn(
            n_state=n_state,
            n_action=n_action,
            seed=seed
        ).to(device)
        
        self.memory = buffer(
            n_action=n_action,
            n_buffer=int(1e5),
            n_batch=64,
            seed=seed
        )
        
        self.t_step = 0
        
    def step(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        
        pass
    
    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnn_hat.eval()
        with torch.no_grad():
            action_values = self.qnn_hat(state_tensor)
        self.qnn_hat.train()
        
        if np.random.random() > eps:
            # when a random number is larger than epsilon, it will "exploit" the action with the best action. it returns the index for the action space. else it will "explore", and then return a random index for the action space.
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.n_action)
        
    def deploy(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnn_deploy.eval()
        with torch.no_grad():
            action_values = self.qnn_deploy(state_tensor)
        self.qnn_deploy.train()
        
        if np.random.random() > eps:
            # when a random number is larger than epsilon, it will "exploit" the action with the best action. it returns the index for the action space. else it will "explore", and then return a random index for the action space.
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.n_action)
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        
        q_targets_next = self.qnn_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        q_targets = rewards + (gamma * q_targets_next * (1-dones))
        q_expected = self.qnn_hat(states).gather(1, actions)
        
        loss = funcs.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnn_hat, self.qnn_target, tau=1e-3)
        
    def soft_update(self, local_model, target_model, tau):
        # IIR
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
