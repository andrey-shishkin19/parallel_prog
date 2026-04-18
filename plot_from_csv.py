import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('results_mpi.csv')

# Построение графика
plt.figure(figsize=(10, 6))

for processes in sorted(df['Processes'].unique()):
    subset = df[df['Processes'] == processes]
    plt.plot(subset['MatrixSize'], subset['ParallelTime_ms'], 
             marker='o', label=f'processes = {processes}')

plt.xlabel('Размер матрицы')
plt.ylabel('Время выполнения (мс)')
plt.title('MPI: Зависимость времени выполнения от размера матрицы\n(разные цвета — разное число процессов)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Сохранение и показ
plt.savefig('mpi_time_vs_size.png', dpi=150)
plt.show()