import matplotlib.pyplot as plt
import numpy as np

sizes = []
times = []

with open('results.txt', 'r') as f:
    lines = f.readlines()
    
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 7:
                size = int(parts[0])
                total_time = int(parts[-1])
                sizes.append(size)
                times.append(total_time)

plt.figure(figsize=(10, 6))

plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)

plt.xlabel('Размер матрицы (n)', fontsize=12)
plt.ylabel('Общее время (мс)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graph.png', dpi=300)
plt.show()

print(f"График сохранён в файл 'graph.png'")