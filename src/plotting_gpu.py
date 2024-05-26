import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('performance_comparison.csv')


color_map = {size: plt.cm.tab10(i) for i, size in enumerate(df['Image Size'].unique())}


gpu_marker = 'o'  # Circle for GPU times
cpu_marker = 'x'  # Cross for CPU times


plt.figure(figsize=(10, 6))


for image_size in df['Image Size'].unique():
    subset = df[df['Image Size'] == image_size]
    color = color_map[image_size]

    plt.plot(subset['Threads'], subset['GPU Time (s)'], marker=gpu_marker, linestyle='-', color=color, label=f'GPU Time {image_size}')
    plt.plot(subset['Threads'], subset['CPU Time (s)'], marker=cpu_marker, linestyle='-', color=color, label=f'CPU Time {image_size}')

plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Performance Comparison by Threads')
plt.legend()
plt.grid(True)
plt.xticks(df['Threads'].unique())
plt.yscale('log')


plt.show()
