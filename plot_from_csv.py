import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
# Для Windows можно раскомментировать:
# plt.rcParams['font.family'] = 'Microsoft YaHei'

# Читаем CSV файл
df = pd.read_csv('results_parallel.csv')

print("=== Данные из CSV ===")
print(df.to_string())
print("\n")

# Создаем фигуру с несколькими графиками
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ производительности параллельного умножения матриц (OpenMP)', 
             fontsize=16, fontweight='bold')

# 1. График времени выполнения
ax1 = axes[0, 0]
for threads in sorted(df['threads'].unique()):
    data = df[df['threads'] == threads]
    ax1.plot(data['size'], data['time_ms'], 'o-', linewidth=2, markersize=6,
             label=f'{threads} потоков')
ax1.set_xlabel('Размер матрицы (n)', fontsize=11)
ax1.set_ylabel('Время выполнения (мс)', fontsize=11)
ax1.set_title('Время выполнения от размера матрицы', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

# 2. График ускорения (Speedup)
ax2 = axes[0, 1]
for threads in sorted(df['threads'].unique()):
    data = df[df['threads'] == threads]
    ax2.plot(data['size'], data['speedup'], 'o-', linewidth=2, markersize=6,
             label=f'{threads} потоков')
# Линия идеального ускорения
for threads in sorted(df['threads'].unique()):
    ax2.axhline(y=threads, color='r', linestyle='--', alpha=0.3)
ax2.set_xlabel('Размер матрицы (n)', fontsize=11)
ax2.set_ylabel('Ускорение (Speedup)', fontsize=11)
ax2.set_title('Ускорение относительно последовательной версии', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. График эффективности
ax3 = axes[1, 0]
for threads in sorted(df['threads'].unique()):
    data = df[df['threads'] == threads]
    ax3.plot(data['size'], data['efficiency'], 'o-', linewidth=2, markersize=6,
             label=f'{threads} потоков')
ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% эффективность')
ax3.set_xlabel('Размер матрицы (n)', fontsize=11)
ax3.set_ylabel('Эффективность (%)', fontsize=11)
ax3.set_title('Эффективность параллелизации', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Сравнение ускорения для разных размеров
ax4 = axes[1, 1]
for size in sorted(df['size'].unique()):
    data = df[df['size'] == size]
    ax4.plot(data['threads'], data['speedup'], 'o-', linewidth=2, markersize=6,
             label=f'Размер {size}x{size}')
# Идеальное ускорение
max_threads = df['threads'].max()
ax4.plot([1, max_threads], [1, max_threads], 'r--', alpha=0.5, 
         label='Идеальное ускорение')
ax4.set_xlabel('Количество потоков', fontsize=11)
ax4.set_ylabel('Ускорение (Speedup)', fontsize=11)
ax4.set_title('Зависимость ускорения от количества потоков', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left')

plt.tight_layout()
plt.savefig('openmp_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Вывод статистики
print("\n=== СТАТИСТИКА ===")
print("-" * 50)
for threads in sorted(df['threads'].unique()):
    data = df[df['threads'] == threads]
    avg_speedup = data['speedup'].mean()
    max_speedup = data['speedup'].max()
    max_speedup_size = data[data['speedup'] == max_speedup]['size'].values[0]
    avg_efficiency = data['efficiency'].mean()
    
    print(f"\nПотоков: {threads}")
    print(f"  Среднее ускорение: {avg_speedup:.2f}x")
    print(f"  Максимальное ускорение: {max_speedup:.2f}x (размер {max_speedup_size})")
    print(f"  Средняя эффективность: {avg_efficiency:.1f}%")

print("\n" + "=" * 50)
print("График сохранен в файл: openmp_results.png")