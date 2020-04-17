import casadi as ca
import numpy as np

# parameters to be imported by all modules
N    = 5
mu   = 0.55 
Tf   = 3.0
Tp   = 1.0
Ts   = 0.0004

umax = 2.0
Q = 0.1*np.array([[1.0, 0.0],[0.0, 1.0]])
R = 0.1*np.array(1.0)
alpha = 0.001

x01 = np.array([0.85, 0.1])
x02 = np.array([-0.7, 0.3])
x01 = np.array([0.85, 0.08])
x03 = np.array([-0.6, -0.6])
x04 = np.array([0.8, -0.2])
x05 = np.array([-0.1, -0.8])
x06 = np.array([-0.9, -0.9])
x07 = np.array([0.90, 0.15])
x08 = np.array([-0.9, 0.8])
x09 = np.array([-0.5, 0.6])

x0_v = [x01, x02, x04, x05, x06]

def ode(x, u):
    xdot = ca.vertcat(\
        x[1] + u * (mu + (1 - mu) * x[0]),
        x[0] + u * (mu - 4 * (1 - mu) * x[1]))

    return xdot
