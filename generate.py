import numpy as np

n = int(input("Введите размер матрицы (но создано будет две): "))

# Генерируем матрицы
A = np.random.randint(0, 10, (n, n))
B = np.random.randint(0, 10, (n, n))

# Сохраняем
with open(f"A{n}.txt", 'w') as f:
    f.write(f"{n}\n")
    for row in A:
        f.write(' '.join(map(str, row)) + '\n')

with open(f"B{n}.txt", 'w') as f:
    f.write(f"{n}\n")
    for row in B:
        f.write(' '.join(map(str, row)) + '\n')

print(f"Созданы файлы A{n}.txt и B{n}.txt")