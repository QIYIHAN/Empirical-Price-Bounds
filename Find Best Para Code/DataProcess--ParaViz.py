import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import pandas as pd
import os

csv_folder = "best_para_output"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
df_all = pd.DataFrame()
for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    df = pd.read_csv(file_path)
    df_all = pd.concat([df_all, df], ignore_index=True)


# n 和 N 对 runtime 的交互效应
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_all, x='n', y='runtime', hue='N', palette='viridis', legend=False)
plt.title('n vs Runtime (colored by N)')

N_values = df_all['N'].unique()  
colors = sns.color_palette('viridis', len(N_values)) 
for i, N_val in enumerate(N_values):
    plt.plot([], [], color=colors[i], label=f'N = {N_val}')
plt.legend(title='N')

plt.show()


# n 和 N 对 iterations 的交互效应
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_all, x='n', y='iterations', hue='N', palette='viridis', legend=False)
plt.title('n vs Iterations (colored by N)')

N_values = df_all['N'].unique()  
colors = sns.color_palette('viridis', len(N_values)) 
for i, N_val in enumerate(N_values):
    plt.plot([], [], color=colors[i], label=f'N = {N_val}')
plt.legend(title='N')

plt.show()

# n 和 N 对 gap_percentage 的交互效应
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_all, x='n', y='gap_percentage', hue='N', palette='viridis', legend=False)
plt.title('n vs Gap Percentage (colored by N)')

N_values = df_all['N'].unique()  
colors = sns.color_palette('viridis', len(N_values)) 
for i, N_val in enumerate(N_values):
    plt.plot([], [], color=colors[i], label=f'N = {N_val}')
plt.legend(title='N')

plt.show()


# N 和 n 对 gap_percentage 的影响
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_all, x='N', y='gap_percentage', hue='n', palette='viridis', legend=False)
plt.title('N vs Gap Percentage (colored by n)')

n_values = list(range(10, 70, 5))
N_values = df_all['n'].unique()  
colors = sns.color_palette('viridis', len(n_values)) 
for i, n_val in enumerate(n_values):
    plt.plot([], [], color=colors[i], label=f'n = {n_val}')
plt.legend(title='n')

plt.show()



