import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dynamics_library import Integrator as inte, Simulation2D as sim2D, RobotUtils as util

print("")


x0 = np.array([1,0,1.0,2.0])
u = np.array([0,0])

t_step = 1e-3

q0_all = []
q1_all = []
t_all = []

t = 0
t_lim = 10
sample_factor = 10

def f_double_integrator(x, u=np.zeros(2)):
    # x' = Ax + Bu
    A = np.array([[0,0,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]])
    B = np.array([[0,0],[0,0],[1,0],[0,1]])
    
    return A @ x + B @ u

def draw_anime(success):
    print('INTEGRATION END')
    print('TIME NOW: ', t)
    print()
    if success:
        print('SYSTEM INTEGRATION SUCCEEDED...')
        save_name = "double_integrator_dynamics"
    else:
        print('SYSTEM INTEGRATION FAILED...')
        save_name = "double_integrator_dynamics" + "_failed"
    
    sim2D().anime(
        t=t_all[::sample_factor], 
        x_states=[
            q0_all[::sample_factor], 
            q1_all[::sample_factor]
        ], 
        ms=1000 * t_step * sample_factor,
        mission="Double Integrator", 
        sim_object="ball",
        sim_info={'ground':0},
        save=False,
        save_name=save_name
    )
    exit()

x_rk4 = x0

while True:

    x_rk4_new = inte().rk4(f_double_integrator, x=x_rk4, u=0, h=t_step, ctrl_on=False)
    
    q0_all.append(x_rk4_new[0])
    q1_all.append(x_rk4_new[1])
    
    t = t + t_step
    t_all.append(t)

    x_rk4 = x_rk4_new
    
    # print(np.abs(theta0_current - q0_ref))
    if t > t_lim:
        break
    
draw_anime(True)