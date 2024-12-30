import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from dynamics_library import Integrator as inte, Simulation2D as sim2D, RobotUtils as util

print("")
# ctrller = "PID" # cascaded pid
ctrller = "SMC" # sliding mode control
# ctrller = "BSTP" # backstepping

x0 = np.array([1,0,0,0])
u = np.array([0,0])
xf = np.array([4,4,0,0])

t_step = 1e-3

q0_all = []
q1_all = []
t_all = []

t = 0
t_lim = 10
sample_factor = 10

# smc shit

# bstp shit

def f_double_integrator(x, u=np.zeros(2)):
    # x' = Ax + Bu
    A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    B = np.array([[0,0],[0,0],[1,0],[0,1]])
    
    return A @ x + B @ u

# pid shit
inte_pos = np.array([0,0])
inte_vel = np.array([0,0])

prerror_pos = np.array([0,0])
prerror_vel = np.array([0,0])

# print(x0)
starto = False

def ctrl_pid(x,xf):
    
    global inte_pos, prerror_pos, inte_vel, prerror_vel
    global starto

    p_now = x[0:2]
    p_tgt = xf[0:2]
    error_pos = p_tgt - p_now
    if not starto:
        prerror_pos = error_pos
    
    Kp_p = 2.0
    Kd_p = 1.5
    Ki_p = 0.1
    
    inte_pos = inte_pos + error_pos * t_step
    vel_ref = Kp_p * error_pos + Kd_p * (error_pos - prerror_pos) / t_step + Ki_p * inte_pos
    prerror_pos = error_pos
    
    v_now = x[2:4]
    error_vel = vel_ref - v_now
    
    if not starto:
        prerror_vel = error_vel
        starto = True
    
    Kp_v = 2.0
    Kd_v = 0.5
    Ki_v = 0.4
    
    inte_vel = inte_vel + error_vel * t_step
    u = Kp_v * error_vel + Kd_v * (error_vel - prerror_vel) / t_step + Ki_v * inte_vel
    prerror_vel = error_vel

    return u

def ctrl_smc(x,xf):    
    C = np.array([[1,0,1,0],[0,1,0,1]])
    A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    B = np.array([[0,0],[0,0],[1,0],[0,1]])
    eta = 2
    
    f = A @ x
    g = B
    
    s = C @ (xf - x)
    h = - eta * np.sign(s)
    u = np.linalg.inv(C @ g) @ (- C @ f - h)
    
    return u

def ctrl_bstp():
    return

def draw_anime(success):
    print('INTEGRATION END')
    print('TIME NOW: ', t)
    print()
    if success:
        print('SYSTEM INTEGRATION SUCCEEDED...')
        save_name = "double_integrator_smc"
    else:
        print('SYSTEM INTEGRATION FAILED...')
        save_name = "double_integrator_smc" + "_failed"
    
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
ind = 0
while True:
    ind = ind + 1
    # if ind == 10:
    #     exit()
    if ctrller == "PID":
        u_input = ctrl_pid(x_rk4, xf=xf)
    elif ctrller == "SMC":
        u_input = ctrl_smc(x_rk4, xf=xf)
    # elif ctrller == "BSTP":
        # u_input = ctrl_pid(x_rk4, xf=xf)
    else:
        print("PICK CTRLLER!")
        exit()
        
    x_rk4_new = inte().rk4(f_double_integrator, x=x_rk4, u=u_input, h=t_step, ctrl_on=True)
    
    q0_all.append(x_rk4_new[0])
    q1_all.append(x_rk4_new[1])
    
    t = t + t_step
    t_all.append(t)

    x_rk4 = x_rk4_new
    
    if t > t_lim:
        break
    
draw_anime(True)