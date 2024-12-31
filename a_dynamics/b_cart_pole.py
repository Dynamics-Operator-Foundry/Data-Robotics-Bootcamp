import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dynamics_library import Integrator as inte, Simulation2D as sim2D, RobotUtils as util

print("")
# ctrller = "PID" # cascaded pid
ctrller = "SMC" # sliding mode control
# ctrller = "BSTP" # backstepping

x0 = np.array([1,0,0.1,2.0]) # position of cart, angle of pole, velo of cart, rate of pole

t_step = 1e-3

p0_all = []
lx0_all = []
lx1_all = []

t_all = []

t = 0
t_lim = 10
sample_factor = 10

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

# pid shit
inte_pos = np.array([0,0])
inte_vel = np.array([0,0])

prerror_pos = np.array([0,0])
prerror_vel = np.array([0,0])

# print(x0)
starto = False


def draw_anime(success):
    print('INTEGRATION END')
    print('TIME NOW: ', t)
    print()
    if success:
        print('SYSTEM INTEGRATION SUCCEEDED...')
        save_name = "cart_pole"
    else:
        print('SYSTEM INTEGRATION FAILED...')
        save_name = "cart_pole" + "_failed"
    
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
        save=False,
        save_name=save_name
    )
    exit()

x_rk4 = x0
ind = 0
while True:
    ind = ind + 1
        
    x_rk4_new = inte().rk4(f_cart_pole, x=x_rk4, h=t_step, ctrl_on=True)
    
    p0_all.append(x_rk4[0])
    lx0_all.append(x_rk4[0] + lp * np.cos(np.pi/2 - x_rk4[1]))
    lx1_all.append(0 + lp * np.sin(np.pi/2 - x_rk4[1]))
    
    t = t + t_step
    t_all.append(t)

    x_rk4 = x_rk4_new
    
    if t > t_lim:
        break
    
draw_anime(True)