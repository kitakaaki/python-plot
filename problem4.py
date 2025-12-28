"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

G = 9.8   # acceleration due to gravity, in m/s^2
# Global variables for parameters, will be updated for each case
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..60 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 60, dt)

plt.figure(figsize=(18, 6))

# (a) m1 = m2 = 1.0 kg, l1 = l2 = 1.0 m, th1 = 60, th2 = 0
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
th1, th2 = 60.0, 0.0
w1, w2 = 0.0, 0.0
state = np.radians([th1, w1, th2, w2])

y = integrate.odeint(derivs, state, t)
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

ax1 = plt.subplot(1, 3, 1)
ax1.plot(x2, y2, linewidth=0.5)
ax1.set_title('(a) m1=m2=1, l1=l2=1, th1=60')
ax1.set_xlabel('x (m)\n© 2025 Huang Yu Chien. All rights reserved.')
ax1.set_ylabel('y (m)')
ax1.grid(True)
ax1.axis('equal')

# (b) m1 = m2 = 1.0 kg, l1 = l2 = 1.0 m, th1 = 60.01, th2 = 0
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
th1, th2 = 60.01, 0.0
w1, w2 = 0.0, 0.0
state = np.radians([th1, w1, th2, w2])

y = integrate.odeint(derivs, state, t)
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

ax2 = plt.subplot(1, 3, 2)
ax2.plot(x2, y2, linewidth=0.5)
ax2.set_title('(b) th1=60.01 (Sensitivity to Initial Conditions)')
ax2.set_xlabel('x (m)\n© 2025 Huang Yu Chien. All rights reserved.')
ax2.set_ylabel('y (m)')
ax2.grid(True)
ax2.axis('equal')

# (c) m1 = 1.0 kg, m2 = 0.5 kg, l1 = 1.0 m, l2 = 0.5 m, th1 = 60, th2 = 0
L1, L2 = 1.0, 0.5
M1, M2 = 1.0, 0.5
th1, th2 = 60.0, 0.0
w1, w2 = 0.0, 0.0
state = np.radians([th1, w1, th2, w2])

y = integrate.odeint(derivs, state, t)
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

ax3 = plt.subplot(1, 3, 3)
ax3.plot(x2, y2, linewidth=0.5)
ax3.set_title('(c) m2=0.5, l2=0.5')
ax3.set_xlabel('x (m)\n© 2025 Huang Yu Chien. All rights reserved.')
ax3.set_ylabel('y (m)')
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.show()