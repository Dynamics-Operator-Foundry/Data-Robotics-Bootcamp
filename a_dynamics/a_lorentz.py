import matplotlib.pyplot as plt
import numpy as np

def rk4(func, x, h):
    
    k1 = func(x) * h
    k2 = func(x + k1 / 2) * h
    k3 = func(x + k2 / 2) * h  
    k4 = func(x + k3) * h
            
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def f_lorentz(x):
    
    sigma = 10
    pho = 28
    beta = 8/3
    
    return np.array([sigma * (x[1] - x[0]), x[0] * (pho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])

x0_all = []
x1_all = []
x2_all = []

z0_all = []
z1_all = []
z2_all = []

t = 0
dt = 1e-3

x = np.array([0.1,0,0])
z = np.array([0.1+1e-3,0,0])

while True:
    x0_all.append(x[0])
    x1_all.append(x[1])
    x2_all.append(x[2])
    z0_all.append(z[0])
    z1_all.append(z[1])
    z2_all.append(z[2])
    
    x = rk4(func=f_lorentz, x=x, h=dt)
    z = rk4(func=f_lorentz, x=z, h=dt)
    
    t = t + dt
    if t > 100.0:
        break


# 3D plot of the Lorenz attractor
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x0_all, x1_all, x2_all, color='blue', lw=0.6, alpha=0.7, label="Trajectory 1")
ax.plot(z0_all, z1_all, z2_all, color='red', lw=0.6, alpha=0.7, label="Trajectory 2")

# Labels and title
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Lorenz Attractor - Butterfly Pattern")

# Save the plot as PNG
plt.savefig("./viz/lorenz_attractor.png", format="png", dpi=300)

plt.show()
    
