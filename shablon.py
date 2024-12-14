import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и ее производную
def func(x):
    return (x - 5)**2

def grad(x):
    return 2 * (x - 5)

# Параметры градиентного спуска
learning_rate = 0.001
num_iterations = 50
x_initial = -14  # Начальное значение

# Градиентный спуск
x_values = [x_initial]
for i in range(num_iterations):
    x_new = x_values[-1] - learning_rate * grad(x_values[-1])
    x_values.append(x_new)


print(f"Минимум достигается при x = {x_values[-1]:.4f}")
print(f"Значение функции в точке минимума = {func(x_values[-1]):.4f}")


x = np.linspace(-15, 25)
y = func(x)

plt.figure(figsize=(5, 5))
plt.plot(x, y, label='f(x) = (x - 5)^2')
plt.scatter(x_values, [func(x) for x in x_values], color='red', label='Gradient Descent', zorder=5)
plt.plot(x_values, [func(x) for x in x_values], color='red', linestyle='dotted', zorder=5)
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
