import numpy as np
import matplotlib.pyplot as plt

# Define the grid
# Interval [-5, 5] with step 0.5
X = np.arange(-5, 5.5, 0.5) 
Y = np.arange(-5, 5.5, 0.5) 
x, y = np.meshgrid(X, Y) 

def plot_field(ax, dx, dy, title):
    # Normalize the arrows
    norm = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dx_n = dx / norm
        dy_n = dy / norm
    
    # Handle cases where norm is 0
    dx_n[norm == 0] = 0
    dy_n[norm == 0] = 0

    ax.quiver(x, y, dx_n, dy_n)
    ax.set_title(title)
    ax.set_xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
    ax.set_ylabel('y')
    ax.grid(True)

plt.figure(figsize=(14, 10))

# (a) dy/dx = x + y
dx = np.ones_like(x)
dy = x + y
ax1 = plt.subplot(2, 2, 1)
plot_field(ax1, dx, dy, r'(a) $dy/dx = x + y$')

# (b) dy/dx = x - y
dx = np.ones_like(x)
dy = x - y
ax2 = plt.subplot(2, 2, 2)
plot_field(ax2, dx, dy, r'(b) $dy/dx = x - y$')

# (c) dy/dx = xy
dx = np.ones_like(x)
dy = x * y
ax3 = plt.subplot(2, 2, 3)
plot_field(ax3, dx, dy, r'(c) $dy/dx = xy$')

# (d) dy/dx = sin(x)cos(y)
dx = np.ones_like(x)
dy = np.sin(x) * np.cos(y)
ax4 = plt.subplot(2, 2, 4)
plot_field(ax4, dx, dy, r'(d) $dy/dx = \sin(x)\cos(y)$')

plt.tight_layout()
plt.show()