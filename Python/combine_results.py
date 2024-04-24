import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tabulate import tabulate
import tkinter as tk
from tkinter import ttk, scrolledtext

def read_results_csv(filename):
    results = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            try:
                # Convert string values to appropriate types
                size = int(row[1])
                a_value = float(row[2])
                b_value = float(row[3])
                execution_time = float(row[4])
                results.append((size, a_value, b_value, execution_time))
            except ValueError:
                print(f"Skipping row: {row}")
    return results

# Read results from numpy, interpreter, CUDA, NumPy optimized, Optimized, Java, and more
numpy_results = read_results_csv('numpy_results.csv')
interpreter_results = read_results_csv('interpreter_results.csv')
cuda_results = read_results_csv('Cuda.csv')
numpy_opt_results = read_results_csv('numpy_OPT_results.csv')
optimized_results = read_results_csv('optimized_results.csv')
java_results = read_results_csv('Java.csv')
more_results = read_results_csv('more.csv')

# Plotting combined results
def plot_combined_results():
    fig_combined, ax_combined = plt.subplots()

    for results, title, color in zip([interpreter_results, optimized_results, numpy_results, numpy_opt_results, java_results, more_results, cuda_results],
                                     ['Interpreter Results', 'Optimized Results', 'NumPy Results', 'NumPy Optimized Results', 'Java Results', 'More Results', 'Cuda Results'],
                                     ['orange', 'purple', 'blue', 'red', 'brown', 'magenta', 'green']):
        sizes, times = zip(*[(size, time) for size, _, _, time in results])
        ax_combined.plot(sizes, times, linestyle='-', marker='o', label=title, color=color)

    ax_combined.set_xlabel('Matrix Size')
    ax_combined.set_ylabel('Execution Time (milliseconds)')
    ax_combined.set_title('Combined Results')
    ax_combined.legend()
    ax_combined.grid(True)

    return fig_combined


# Plotting individual results without the blank subplot
fig_individual, axs_individual = plt.subplots(4, 2, figsize=(15, 15))

# Adjusting vertical spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Colors for individual graphs
colors = ['orange', 'purple', 'blue', 'red', 'brown', 'magenta', 'green']

# Plotting Interpreter and Optimized graphs side-by-side
for results, title, color, ax in zip([interpreter_results, optimized_results],
                                     ['Interpreter Results', 'Optimized Results'],
                                     ['orange', 'purple'],
                                     axs_individual[0, :]):
    data_dict = {}
    for (size, a_val, b_val, time) in results:
        if (a_val, b_val) not in data_dict:
            data_dict[(a_val, b_val)] = ([], [])
        data_dict[(a_val, b_val)][0].append(size)
        data_dict[(a_val, b_val)][1].append(time)
    for (a_val, b_val), (sizes, times) in data_dict.items():
        if len(sizes) >= 2:
            ax.plot(sizes, times, linestyle='-', marker='o', label=title, color=color)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Execution Time (milliseconds)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Plotting NumPy and NumPy Optimized graphs side-by-side
for results, title, color, ax in zip([numpy_results, numpy_opt_results],
                                     ['NumPy Results', 'NumPy Optimized Results'],
                                     ['blue', 'red'],
                                     axs_individual[1, :]):
    data_dict = {}
    for (size, a_val, b_val, time) in results:
        if (a_val, b_val) not in data_dict:
            data_dict[(a_val, b_val)] = ([], [])
        data_dict[(a_val, b_val)][0].append(size)
        data_dict[(a_val, b_val)][1].append(time)
    for (a_val, b_val), (sizes, times) in data_dict.items():
        if len(sizes) >= 2:
            ax.plot(sizes, times, linestyle='-', marker='o', label=title, color=color)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Execution Time (milliseconds)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Plotting Java and More graphs side-by-side
for results, title, color, ax in zip([java_results, more_results],
                                     ['Java Results', 'More Results'],
                                     ['brown', 'magenta'],
                                     axs_individual[2, :]):
    data_dict = {}
    for (size, a_val, b_val, time) in results:
        if (a_val, b_val) not in data_dict:
            data_dict[(a_val, b_val)] = ([], [])
        data_dict[(a_val, b_val)][0].append(size)
        data_dict[(a_val, b_val)][1].append(time)
    for (a_val, b_val), (sizes, times) in data_dict.items():
        if len(sizes) >= 2:
            ax.plot(sizes, times, linestyle='-', marker='o', label=title, color=color)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Execution Time (milliseconds)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Centering the Cuda Results graph
axs_individual[3, 0].remove()  # Remove the empty subplot
axs_individual[3, 1].remove()  # Remove the empty subplot

# Determine the position for Cuda Results
ax = fig_individual.add_subplot(4, 2, (7, 8))  # Adjusted position to cover two columns and be centered vertically

sizes, times = zip(*[(size, time) for size, _, _, time in cuda_results])
ax.plot(sizes, times, linestyle='-', marker='o', label='Cuda Results', color='green')
ax.set_xlabel('Matrix Size')
ax.set_ylabel('Execution Time (milliseconds)')
ax.set_title('Cuda Results')
ax.legend()
ax.grid(True)

# Adjust layout to ensure correct spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusted to avoid covering the bottom margin



# Display the plots and tables in a single window with tabs
root = tk.Tk()
root.title("Matrix Multiplication Benchmark Results")

# Create a notebook (tabs)
notebook = ttk.Notebook(root)

# Add tab for combined results plot
combined_tab = tk.Frame(notebook)
notebook.add(combined_tab, text='Combined Plot')
combined_canvas = FigureCanvasTkAgg(plot_combined_results(), master=combined_tab)
combined_canvas.draw()
combined_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Add tab for individual results plots
individual_tab = tk.Frame(notebook)
notebook.add(individual_tab, text='Individual Plots')
individual_canvas = FigureCanvasTkAgg(fig_individual, master=individual_tab)
individual_canvas.draw()
individual_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Add tab for tables
table_tab = tk.Frame(notebook)
notebook.add(table_tab, text='Tables')

# Constructing the tables with two tables per line
tables = [
    (interpreter_results, 'Interpreter Results'),
    (optimized_results, 'Optimized Results'),
    (numpy_results, 'NumPy Results'),
    (numpy_opt_results, 'NumPy Optimized Results'),
    (java_results, 'Java Results'),
    (more_results, 'More Results'),
    (cuda_results, 'Cuda Results')
]

for i in range(0, len(tables), 2):
    frame = tk.Frame(table_tab)
    frame.pack(side=tk.TOP, fill='both', expand=True)

    for j in range(2):
        if i + j < len(tables):
            results, title = tables[i + j]
            table_text = scrolledtext.ScrolledText(frame, width=60, height=10)
            table_text.pack(side=tk.LEFT, fill='both', expand=True)
            table_data = tabulate([["", size, round(a_val, 4), round(b_val, 4), round(time, 4)] for (size, a_val, b_val, time) in results], headers=["", "Size", "A Value", "B Value", "Execution Time (milliseconds)"], tablefmt='pretty')
            table_text.insert(tk.END, title + '\n\n' + table_data)

notebook.pack(expand=True, fill='both')

# Running the tkinter event loop
root.mainloop()
