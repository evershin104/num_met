import numpy as np
from scipy.linalg import solve, inv
from matplotlib import pyplot as plt


np.set_printoptions(precision=11)
A = np.matrix([[1.362, 0.432, -0.599, 0.202],
               [0.202, 1.362, 0.432, -0.599],
               [-0.599, 0.202, 1.362, 0.432],
               [0.432, -0.599, 0.202, 1.362]])

b = np.array([1.941, -0.230, -1.941, 0.230])
x = solve(A, b)
x0 = np.array([1.941, -0.230, -1.941, 0.230])
D = np.diag(np.diag(A))


# Проверка на свойство диагонального преобладания
def diag_dom(A):
    for i in range(0, A.shape[0]):
        if A[i, i] < sum([abs(A[i, j]) / 2 for j in range(0, A.shape[0]) if i != j]):
            return False
    print('Достаточное условие сходимости выполнено')
    return True


def iterate(x_k):
    x_next = np.array([])
    for i in range(0, np.size(x_k)):
        summa = (sum([A[i, j] * x_k[j]
                 for j in range(0, np.size(x_k))])) - b[i]
        value = x_k[i] - 1 / A[i, i] * summa
        x_next = np.append(x_next, value)
    return x_next


print(f'Норма матрицы перехода итер. метода (l1) = {np.linalg.norm(np.eye(A.shape[0]) - inv(D) * A, ord = 1)}')
print(f'Норма матрицы перехода итер. метода (l2) = {np.linalg.norm(np.eye(A.shape[0]) - inv(D) * A, ord = 2)}')
print(f'Норма матрицы перехода итер. метода (l_inf) = {np.linalg.norm(np.eye(A.shape[0]) - inv(D) * A, ord = np.inf)}')
it = 1
diff = 200
if not diag_dom(A):
    print('Достаточное условие сходимости не выполнено')

x1_eps = np.array([])
x2_eps = np.array([])
x3_eps = np.array([])
x4_eps = np.array([])
r = np.array([])
while diff > pow(10, -11):
    xn = iterate(x0)
    diff = np.max(np.abs(xn - x0))
    x0 = xn
    print(f'{it + 1}. {xn}')
    x1_eps = np.append(x1_eps, abs(xn[0] - x[0]))
    x2_eps = np.append(x2_eps, abs(xn[1] - x[1]))
    x3_eps = np.append(x3_eps, abs(xn[2] - x[2]))
    x4_eps = np.append(x4_eps, abs(xn[3] - x[3]))
    r = np.append(r, np.sum(np.abs((A * xn.reshape(4, 1) - b.reshape(4, 1)).reshape(1, 4))))
    it += 1

root = xn
print(f'SciPy.Linalg solution (0.0024784460000000005 sec.):'
      f'\n{x}\nJacobi solution (0.0034776419999999995 sec.):\n{xn}'
      f'\nНевязка на последней итерации = {r[it - 2]}')

plt.plot(np.arange(0, it - 1), x3_eps, color = 'blue', label = r'$x_{3}^{eps}, x_{1}^{eps}$')
plt.plot(np.arange(0, it - 1), x4_eps, color = 'red', label = r'$x_{4}^{eps}, x_{2}^{eps}$')
plt.plot(np.arange(0, it - 1), r, color = 'green', label = r'$r^{(k)} = \sum|Ax^{(k)} - b|$')
plt.hlines(0, 0, it - 2, color = 'black', linestyles = '--')
plt.legend()
plt.show()
