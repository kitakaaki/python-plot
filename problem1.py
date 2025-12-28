import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives
# f(x) = x^2 - 2x + 5
def f(x):
    return x**2 - 2*x + 5

# f'(x) = 2x - 2
def df(x):
    return 2*x - 2

# (a) Gradient Descent
x0 = 0.0
alpha_gd = 0.1
n_iterations = 10

x_gd = [x0]
y_gd = [f(x0)]

print("Gradient Descent:")
print(f"n=0, x={x0:.4f}, y={f(x0):.4f}")

x_curr = x0
for i in range(n_iterations):
    # Gradient Descent update: x_new = x_old - alpha * f'(x_old)
    x_next = x_curr - alpha_gd * df(x_curr)
    
    x_gd.append(x_next)
    y_gd.append(f(x_next))
    x_curr = x_next
    print(f"n={i+1}, x={x_curr:.4f}, y={f(x_curr):.4f}")

# (b) Newton's Method (Optimization)
# Update rule: x_new = x_old - alpha * f'(x_old) / f''(x_old)
x0 = 0.0
alpha_nm = 1.0

x_nm = [x0]
y_nm = [f(x0)]

print("\nNewton's Method:")
print(f"n=0, x={x0:.4f}, y={f(x0):.4f}")

x_curr = x0
for i in range(n_iterations):
    # Newton's method for optimization
    # x_new = x - alpha * (f(x) / f'(x))
    update = f(x_curr) / df(x_curr)
    x_next = x_curr - alpha_nm * update
    
    x_nm.append(x_next)
    y_nm.append(f(x_next))
    x_curr = x_next
    print(f"n={i+1}, x={x_curr:.4f}, y={f(x_curr):.4f}")

# (c) Plotting
# Range -0.5 to 2.5
x_plot = np.linspace(-0.5, 2.5, 100)
y_plot = f(x_plot)

plt.figure(figsize=(12, 5))

# Plot 1: Gradient Descent
plt.subplot(1, 2, 1) 
plt.plot(x_plot, y_plot, label='f(x)')
plt.plot(x_gd, y_gd, 'o-', color='red', label='Gradient Descent')
for i, (xi, yi) in enumerate(zip(x_gd, y_gd)):
    plt.text(xi, yi, f' n{i}', fontsize=9)
plt.title('Gradient Descent')
plt.xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Plot 2: Newton's Method
plt.subplot(1, 2, 2)
plt.plot(x_plot, y_plot, label='f(x)')
plt.plot(x_nm, y_nm, 'o-', color='green', label="Newton's Method")
for i, (xi, yi) in enumerate(zip(x_nm, y_nm)):
    plt.text(xi, yi, f' n{i}', fontsize=9)
plt.title("Newton's Method")
plt.xlabel('x\n© 2025 Huang Yu Chien. All rights reserved.')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()