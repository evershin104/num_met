import numpy as np

np.set_printoptions(precision=8)


def f(i, j, matrix):
    if matrix[i, i] != matrix[j, j]:
        return 0.5 * np.arctan((2 * matrix[i, j]) / (matrix[i, i] - matrix[j, j]))
    else:
        return np.pi / 4


def c(i, j, matrix): return np.cos(f(i, j, matrix))


def s(i, j, matrix): return np.sin(f(i, j, matrix))


def t(matrix): return sum([np.power(matrix[i, j], 2) for i in range(0, 4) for j in range(0, 4) if i != j])


def findMaxInd(matrix):
    # Обнулим все значения матрицы, которые лежат на главной диагонали и ниже
    dup = np.copy(matrix)
    i, j = np.mgrid[0:4, 0:4]
    dup[i >= j] = 0
    # Получим индекс максимального по модулю элемента урезанной матрицы
    i, j = np.where(np.abs(dup) == np.max(np.abs(dup)))
    return i[0], j[0]


def iterate(matrix, j_rot):
    U = np.eye(4)
    i, j = findMaxInd(matrix)
    U[i, i] = c(i, j, matrix)
    U[j, j] = c(i, j, matrix)
    U[i, j] = -s(i, j, matrix)
    U[j, i] = s(i, j, matrix)
    matrix = U.T * matrix * U
    j_rot = j_rot * U
    return matrix, j_rot, U


def find_eig_values(matrix):
    V = np.eye(4)
    eig_vectors_approx = np.eye(4)
    while t(matrix) > 1e-6:
        matrix, V, new_rotation = iterate(matrix, V)
        eig_vectors_approx = eig_vectors_approx * np.asmatrix(new_rotation)
    return np.squeeze(np.asarray(matrix.diagonal())), eig_vectors_approx


def calc_discrepancy(matrix, eig_vector, l):
    return matrix * eig_vector - l * eig_vector


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


print(f'Eigenvalues NumPy = {np.linalg.eigvals(A)}')

eigenvalues, eigenvectors = find_eig_values(A)
print(f'Eigenvalues = {eigenvalues}\n')
print(f'Eigenvectors as columns of this matrix:\n{eigenvectors}\n')
for i in range(0, np.shape(A)[0]):
    print(f'Discrepancy for eigenvalue = {eigenvalues[i]}:')
    print(f'{calc_discrepancy(A, eigenvectors[0:np.shape(A)[0], i], eigenvalues[i])}\n')
