import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funcs
import random
from collections import namedtuple, deque

class ReplayBuffer():
    def __init__(
        self, 
        n_action, 
        n_buffer, 
        n_batch, 
        seed,
        device="cpu"
    ):
        self.n_action = n_action
        self.memory = deque(maxlen=n_buffer)
        self.n_batch = n_batch
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(
            state=state, 
            action=action, 
            reward=reward,
            next_state=next_state,
            done=done
        ))
        
    def sample(self):
        experiences = random.sample(
            self.memory,
            k=self.n_batch
        )
        
        for e in experiences:
            if e is not None:
                states = torch.from_numpy(
                    np.vstack([e.state])
                ).float().to(self.device)
                actions = torch.from_numpy(
                    np.vstack([e.action])
                ).long().to(self.device)
                rewards = torch.from_numpy(
                    np.vstack([e.reward])
                ).float().to(self.device)
                next_states = torch.from_numpy(
                    np.vstack([e.next_state])
                ).float().to(self.device)
                dones = torch.from_numpy(
                    np.vstack([e.done]).astype(np.uint8)
                ).float().to(self.device)
            
        return (
            states,
            actions, 
            rewards,
            next_states,
            dones
        )
        
    def __len__(self):
        return len(self.memory)

class QNN(nn.Module):
    def __init__(
        self,
        n_state,
        n_action,
        seed,
        n_fc1=64,
        n_fc2=64,
        device="cpu"
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.fc3 = nn.Linear(n_fc2, n_action)
        self.to(self.device)

    def forward(self, x):
        y = funcs.relu(self.fc1(x))
        y = funcs.relu(self.fc2(y))
        y = self.fc3(y)
        
        return y

