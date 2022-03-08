"""Степенной метод"""
import numpy as np
# Вывод np значений до 8 знаков после запятой
np.set_printoptions(precision=8)


# Нормализация вектора
def normalize(v):
    sum_ = sum([np.power(i, 2) for i in np.squeeze(np.asarray(v))])
    return np.matrix([i/np.sqrt(sum_) for i in np.squeeze(np.asarray(v))]).T


# Расчет значений следующей итерации
def iterate(matrix, v):
    v_n = matrix * v
    return v_n, matrix * v_n


# Реализация степенного метода
def find_max_abs_eigenvalue(matrix):
    y = normalize(np.matrix([1, 1, 1, 1]).T)
    y, y_next = iterate(matrix, y)
    eig_approx = [y_next[0, 0] / y[0, 0]]
    y, y_next = iterate(matrix, y)
    eig_approx.append(y_next[0, 0] / y[0, 0])
    while abs(eig_approx[-1] - eig_approx[-2]) > 1e-8:
        eig_approx[-2] = eig_approx[-1]
        y, y_next = iterate(matrix, y)
        eig_approx[-1] = y_next[0, 0] / y[0, 0]
    return eig_approx[-1], y_next


def calc_discrepancy(matrix, last_y, l):
    return matrix * last_y - l * last_y


C = np.matrix([[0.05, 0, 0, 0],
               [0, 0.03, 0, 0],
               [0, 0, 0.02, 0],
               [0, 0, 0, 0.04]])
k = 1
D = np.matrix([[1.342, 0.432, -0.599, 0.202],
               [0.432, 1.342, 0.256, -0.599],
               [-0.599, 0.256, 1.342, 0.532],
               [0.202, -0.599, 0.532, 1.342]])
A = D + k * C

eig, y_n = find_max_abs_eigenvalue(A)
print(f'Max. abs. eigenvalue: {eig}')
print(f'NumPy max. abs. eigenvalue: {np.max(np.abs(np.linalg.eigvals(A)))}')
print(f'Discrepancy =\n{calc_discrepancy(A, y_n, eig)}')
