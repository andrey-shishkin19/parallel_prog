import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        m = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            m.append(row)
    return np.array(m)

print("Введите:")
fileA = input("Файл A: ")
fileB = input("Файл B: ")
fileC = input("Файл результата: ")

A = read_matrix(fileA)
B = read_matrix(fileB)
C_cpp = read_matrix(fileC)

C_expected = np.dot(A, B)

if np.allclose(C_cpp, C_expected, atol=1e-9):
    print("\n ВСЁ ПРАВИЛЬНО!")
    print(f"Максимальная ошибка: {np.max(np.abs(C_cpp - C_expected)):.2e}")
else:
    print("\n ОШИБКА!")