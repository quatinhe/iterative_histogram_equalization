import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('results.csv')


pivot_data = data.pivot_table(index=['size', 'iterations'], columns='test_type', values='time_seconds')


pivot_data.reset_index(inplace=True)


plt.figure(figsize=(10, 6))


for size in pivot_data['size'].unique():
    subset = pivot_data[pivot_data['size'] == size]
    plt.plot(subset['iterations'], subset['sequential'], marker='o', linestyle='-', label=f'Sequential {size}')
    plt.plot(subset['iterations'], subset['parallel'], marker='o', linestyle='-', label=f'Parallel {size}')

plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison by Size and Iteration Count')
plt.legend()
plt.grid(True)
plt.show()
