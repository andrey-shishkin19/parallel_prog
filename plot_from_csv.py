import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('results_parallel.csv')

# Построение графика
plt.figure(figsize=(10, 6))

for threads in sorted(df['threads'].unique()):
    subset = df[df['threads'] == threads]
    plt.plot(subset['size'], subset['time_ms'], marker='o', label=f'threads = {threads}')

plt.xlabel('Размер матрицы')
plt.ylabel('Время выполнения (мс)')
plt.title('Зависимость времени выполнения от размера матрицы\n(разные цвета — разное число потоков)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Сохранение и показ
plt.savefig('time_vs_size.png', dpi=150)
plt.show()
