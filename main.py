import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и ее производную
def func(x):
 # return 3 * x[0]**2 + 2 * x[1]**2 + 4 * x[0] * x[1] - 5 * x[0] + 6 * x[1]  тут пишем свою функциюю


def grad(x):
 # return np.array([6 * x[0] + 4 * x[1] - 5, 4 * x[0] + 4 * x[1] + 6]) тут уже пишем ее производную от функции

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


'''
Дана функция f(x) = 3x21+ 2x22+ 4x1x2−5x1+ 6x2- найти минимум функции с помощью метода градиентного спуска
'''
