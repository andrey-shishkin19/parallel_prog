import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

plt.figure(figsize=(10, 6))

for block in df['BlockConfiguration'].unique():
    data = df[df['BlockConfiguration'] == block]
    plt.plot(data['MatrixSize'], data['ParallelTime_ms'], marker='o', linewidth=2, label=f'Блок {block}')

cpu_data = df[['MatrixSize', 'SequentialTime_ms']].drop_duplicates().sort_values('MatrixSize')
plt.plot(cpu_data['MatrixSize'], cpu_data['SequentialTime_ms'], marker='s', linewidth=2, linestyle='--', color='black', label='CPU')

plt.xlabel('Размер матрицы')
plt.ylabel('Время выполнения (мс)')
plt.title('Умножение матриц: CPU vs GPU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('graph.png', dpi=150)
plt.show()