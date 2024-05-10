import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Load the results from the CSV file
data = pd.read_csv('results.csv')

# Aggregate data by averaging time_seconds where iterations are the same
data = data.groupby(['test_type', 'iterations', 'threads']).agg({'time_seconds': 'mean'}).reset_index()

plt.figure(figsize=(12, 8))

# Sequential data processing
sequential_data = data[data['test_type'] == 'sequential']
x_seq = np.array(sequential_data['iterations'].unique())  # Unique iterations
y_seq = np.array(sequential_data.groupby('iterations')['time_seconds'].mean())  # Average time for each iteration

if len(x_seq) > 1:
    x_seq_new = np.linspace(x_seq.min(), x_seq.max(), 300)
    spl = make_interp_spline(x_seq, y_seq, k=3)
    y_seq_smooth = spl(x_seq_new)
    plt.plot(x_seq_new, y_seq_smooth, linestyle='-', color='black', label='Sequential')

# Parallel data processing
colors = ['blue', 'green', 'red']
for i, threads in enumerate(sorted(data['threads'].unique())):
    subset = data[(data['test_type'] == 'parallel') & (data['threads'] == threads)]
    x = np.array(subset['iterations'].unique())
    y = np.array(subset.groupby('iterations')['time_seconds'].mean())

    if len(x) > 1:
        x_new = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        plt.plot(x_new, y_smooth, linestyle='-', color=colors[i % len(colors)], label=f'Parallel {threads} Threads')

plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison by Iteration Count and Thread Number')
plt.legend(title='Configuration')
plt.grid(True)
plt.show()
