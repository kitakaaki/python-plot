import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))

# (a) y = e^(-x) (2cos3x -sin3x)  I : [0, 1]
x_a = np.linspace(0, 1, 100)
y_a = np.exp(-x_a) * (2 * np.cos(3 * x_a) - np.sin(3 * x_a))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(x_a, y_a)
ax1.set_title(r'(a) $y = e^{-x} (2\cos3x -\sin3x)$')
ax1.set_xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
ax1.set_ylabel('y')
ax1.grid(True)

# (b) y = e^(-x) + e^x + 0.5xe^x  I : [0, 1]
x_b = np.linspace(0, 1, 100)
y_b = np.exp(-x_b) + np.exp(x_b) + 0.5 * x_b * np.exp(x_b)

ax2 = plt.subplot(2, 2, 2)
ax2.plot(x_b, y_b)
ax2.set_title(r'(b) $y = e^{-x} + e^x + 0.5xe^x$')
ax2.set_xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
ax2.set_ylabel('y')
ax2.grid(True)

# (c) y = cos2x + 0.25xsin2x  I : [0, 4pi]
x_c = np.linspace(0, 4 * np.pi, 400)
y_c = np.cos(2 * x_c) + 0.25 * x_c * np.sin(2 * x_c)

ax3 = plt.subplot(2, 2, 3)
ax3.plot(x_c, y_c)
ax3.set_title(r'(c) $y = \cos2x + 0.25x\sin2x$')
ax3.set_xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
ax3.set_ylabel('y')
ax3.grid(True)

# (d) y = e^(-x) + 1 + e^(2x) I : [0, 1]
x_d = np.linspace(0, 1, 100)
y_d = np.exp(-x_d) + 1 + np.exp(2 * x_d)

ax4 = plt.subplot(2, 2, 4)
ax4.plot(x_d, y_d)
ax4.set_title(r'(d) $y = e^{-x} + 1 + e^{2x}$')
ax4.set_xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
ax4.set_ylabel('y')
ax4.grid(True)

plt.tight_layout()
plt.show()

