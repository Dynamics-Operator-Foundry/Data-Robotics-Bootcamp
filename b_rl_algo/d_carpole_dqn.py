import sys
import os
import numpy as np
from collections import deque
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dynamics_library import Integrator as inte, Simulation2D as sim2D
from a_dqn import DQN as dqn

print("")
ctrller = "DQN" # Deep Q-Network
# ctrller = "PPO" # Proximal policy optimization
# ctrller = "DDPG" # Deep Deterministic Policy Gradient
# ctrller = "TD3" # Twin-Delayed Deep Deterministic (TD3)


t_step = 1e-2

p0_all = []
lx0_all = []
lx1_all = []

t_all = []

t = 0
t_lim = 5
sample_factor = 1

mp = 1.0 # mass of pole
lp = 1.0 # length of pole
g = 9.81 # gravity constant
mc = 2.0 # mass of cart


def f_cart_pole(x, u=0):
    global mp, lp, g, mc
    
    p = x[0]
    theta = x[1]
    pdot = x[2]
    thetadot = x[3]
    
    pddot = (u + mp * lp * thetadot**2 * np.sin(theta) - mp * g * np.sin(theta) * np.cos(theta)) 
    pddot = pddot / (mc + mp * (1 - np.cos(theta)**2))
    
    # thetaddot = - u * np.cos(theta) - mp * lp * (thetadot**2 * np.sin(theta) * np.cos(theta) + g * np.sin(theta)) 
    thetaddot = (-u * np.cos(theta) + mp * lp * (thetadot**2 * np.sin(theta) * np.cos(theta) + g * np.sin(theta)))

    thetaddot = thetaddot / (lp * (mc + mp * (1 - np.cos(theta)**2)))
    
    return np.array([
        pdot, 
        thetadot, 
        pddot,
        thetaddot
    ])

starto = False


def draw_anime(success):
    print('INTEGRATION END')
    print('TIME NOW: ', t)
    print()
    if success:
        print('SYSTEM INTEGRATION SUCCEEDED...')
        save_name = "cart_pole_balance_dqn"
    else:
        print('SYSTEM INTEGRATION FAILED...')
        save_name = "cart_pole_balance_dqn" + "_failed"
    
    sim2D().anime(
        t=t_all[::sample_factor], 
        x_states=[
            p0_all[::sample_factor], 
            lx0_all[::sample_factor], 
            lx1_all[::sample_factor]
        ], 
        ms=1000 * t_step * sample_factor,
        mission="Cart Pole", 
        sim_object="cart_pole",
        sim_info={'ground':0},
        save=True,
        save_name=save_name
    )
    exit()

def env_reset():
    global p0_all, lx0_all, lx1_all
    p0_all.clear()
    lx0_all.clear()
    lx1_all.clear()
    
    cart_position_range = (-0.05, 0.05)  
    cart_velocity_range = (-0.01, 0.01)  
    pole_angle_range = (-0.1, 0.1) # for balance 
    # pole_angle_range = (-0.05 + np.pi, 0.05 + np.pi) # for swing and balance
    pole_angular_velocity_range = (-0.05, 0.05)
    
    initial_state = np.array([
        np.random.uniform(*cart_position_range),
        np.random.uniform(*pole_angle_range),  
        np.random.uniform(*cart_velocity_range),
        np.random.uniform(*pole_angular_velocity_range)
    ])
    
    # Return the numpy array containing the initial state
    return initial_state


num_episodes = 500
max_steps_per_episode = 500
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_rate = 0.9999
gamma = 0.99
lr = 0.0025
buffer_size = 10000
buffer = deque(maxlen=buffer_size)
batch_size = 256
update_frequency = 10

n_input = 4
n_output = 4 # for balance
# n_output = 8 # for swing and balance
DQN_agent = dqn(
    n_state=n_input,
    n_action=n_output,
    seed=170715,
    lr=lr
)

def anime_buffer(x_rk4):
    global p0_all, lx0_all, lx1_all
    
    p0_all.append(x_rk4[0])
    lx0_all.append(x_rk4[0] + lp * np.cos(np.pi/2 - x_rk4[1]))
    lx1_all.append(0 + lp * np.sin(np.pi/2 - x_rk4[1]))
    
    return

x_rk4 = env_reset()
action_dict = np.array([-10, -5, 5, 10]) # for balance
# action_dict = np.array([-40, -10, -5, -2, 2, 5, 10, 40]) # for swing and balance

train = False
if train:
    for episode in range(num_episodes):
        reward_acc = 0
        x_rk4 = env_reset()
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode)) 
        # this is for exploration 
        
        for step in range(max_steps_per_episode):
            action_ind = DQN_agent.act(state=x_rk4, eps=epsilon)
            u_k = action_dict[action_ind]
            
            x_rk4_new = inte().rk4(
                f_cart_pole, 
                x=x_rk4, 
                u=u_k, 
                h=t_step, 
                ctrl_on=True
            )
            
            # # for swing
            # reward = 10 - (x_rk4_new[1]**2 + 0.1 * x_rk4_new[3]**2 + 0.1 * x_rk4_new[0]**2 + 0.1 * x_rk4_new[1]**2)
            
            # if np.abs(x_rk4_new[1]) > (30 / 180 * np.pi) or np.abs(x_rk4_new[0]) > 3:
            #     reward -= 50
                
            reward = - (x_rk4_new[1]**2 + 0.1 * x_rk4_new[3]**2 + 0.1 * x_rk4_new[0]**2 + 0.001 * u_k**2)
            if np.abs(x_rk4_new[1]) < (0.5 / 180 * np.pi):
                reward += 10

            
            done = False
            # if np.abs(x_rk4_new[0]) > 5:
                # done = True

            # # for balance
            # reward = 1
            # if np.abs(x_rk4_new[1]) > (0.5 / 180 * np.pi):
            #     reward = -1
            
            # done = False
            # if np.abs(x_rk4_new[1]) > (30 / 180 * np.pi):
            #     done = True
            # elif np.abs(x_rk4_new[0]) > 3:
            #     done = True

            reward_acc += reward
            
            buffer.append((x_rk4, action_ind, reward, x_rk4_new, done))
            
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                DQN_agent.learn(batch, gamma)
                
            x_rk4 = x_rk4_new
            anime_buffer(x_rk4)
            
            if done:            
                break
        
        if (episode + 1) % update_frequency == 0:
            # draw_anime(True)
            print(f"Episode {episode + 1}: Finished training, reward: {reward_acc}")
            torch.save(DQN_agent.qnn_hat.state_dict(), f'dqn_weights_episode_{episode + 1}.pt')
            
else:
    
    DQN_agent.qnn_deploy.load_state_dict(torch.load('dqn_weights_episode_450.pt'))
    
    x_rk4 = env_reset()
    ind = 0

    
    while True:
        ind = ind + 1
        
        print(x_rk4)
        action_ind = DQN_agent.deploy(state=x_rk4, eps=0.0)
        u_k = action_dict[action_ind]
        print(action_ind)
        
        # u_k = 0
        
        x_rk4_new = inte().rk4(f_cart_pole, x=x_rk4, u=u_k,h=t_step, ctrl_on=True)
        
        p0_all.append(x_rk4[0])
        lx0_all.append(x_rk4[0] + lp * np.cos(np.pi/2 - x_rk4[1]))
        lx1_all.append(0 + lp * np.sin(np.pi/2 - x_rk4[1]))
        
        t = t + t_step
        t_all.append(t)

        x_rk4 = x_rk4_new
        
        if t > t_lim:
            break

    draw_anime(True)
