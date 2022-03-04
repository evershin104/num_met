import numpy as np
from matplotlib import pyplot as plt


# функция
def f(x):
	return np.log(x) - 1 / np.power(x, 2)


# производная
def f_(x):
	return 1 / x + 2 / np.power(x, 3)


figure, axis = plt.subplots(3)
x = np.linspace(0.5, 5, 100)
axis[0].plot(x, f(x), color = 'red', label = r'$f(x) = ln(x) - \frac{1}{x^2}$')
axis[0].hlines(0, 0, 5, color = 'black')
axis[0].legend()
axis[1].plot(x, f_(x), color = 'red', label = r"$f'(x) = \frac{1}{x} + \frac{2}{x^3}$")
axis[1].hlines(0, 0, 5, color = 'black')
axis[1].legend()

# 0.1
a = 1
b = 2
M = max(abs(f_(np.linspace(a, b, 100))))
m = min(abs(f_(np.linspace(a, b, 100))))
print(f'M = {M}')
print(f'm = {m}')

x_last = a
x_next = a
it = 1
roots = np.array([])
eps = pow(10, -10)
diff = 100
# Для погрешности приближенного решения также справедлива и оценка (7)
# while M / m * abs(x_next - x_last) > eps or abs(f(x_next)) / m > eps:
while diff > eps:
	x_next = x_last - f(x_last) / f_(a)
	diff = abs(x_next - x_last)
	print(f"{it}. {x_next}")
	it += 1
	x_last = x_next
	roots = np.append(roots, x_last)
print(f"Root = {x_next}\nОтносительная погрешность = {eps / x_next}\nАбсолютная погрешность = {eps}")
diff_abs = np.array([abs(roots[i + 1] - roots[i])
                     for i in range(0, np.size(roots) - 1)])
axis[2].plot(np.arange(0, np.size(roots), step = 1), abs(f(roots)) / m,
             color = 'blue',
             label = r'$\frac{|f(x_{n})|}{m}$')
axis[2].plot(np.arange(0, np.size(diff_abs), step = 1), M / m * diff_abs,
             color = 'green',
             label = r'$\frac{M}{m}|x_{n+1} - x_{n}|$')
axis[2].hlines(eps, 0, np.size(diff_abs), color = 'black', label = r'$eps$')
axis[2].legend()
plt.show()
