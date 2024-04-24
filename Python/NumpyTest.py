#This is Matrix Multiplication Benchmarking code in Numpy

import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def matrix_multiply_np(A, B):
    return np.dot(A, B)

def generate_matrix_np(size, value):
    return np.full((size, size), value)

def generate_multiplier_matrix_np(size, value):
    return np.eye(size) * value

def benchmark_matrix_multiplication_np(size):
    A_values = [1, 2, 0]  # All 1s, all 2s, and incrementing by 1
    B_values = [generate_multiplier_matrix_np(size, value) for value in [1, 2, 0]]  # Identity matrix, all 1s, all 2s

    results = []

    for A_value in A_values:
        for B_value in B_values:
            A = generate_matrix_np(size, A_value)
            B = B_value
            start_time = time.time()
            C = matrix_multiply_np(A, B)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            results.append((size, A_value, B_value[0, 0], execution_time))

    return results

# Benchmarking matrix multiplication for sizes 128, 256, and 512
sizes = [128, 256, 512]
all_results_np = []

for size in sizes:
    all_results_np.extend(benchmark_matrix_multiplication_np(size))

# Outputting results as a table in a separate window
root = tk.Tk()
root.title("Matrix Multiplication Benchmark Results")

# Create a frame to hold both the table and the plot
frame = tk.Frame(root)
frame.pack(expand=True, fill='both')

# Create a scrolled text widget for the table
text_area = scrolledtext.ScrolledText(frame, width=80, height=15)
text_area.pack(expand=True, fill='both', side='left')

# Create a canvas for the plot
canvas = tk.Canvas(frame, bg='white')
canvas.pack(expand=True, fill='both', side='right')

# Constructing the table
headers = ["Size", "A Value", "B Value", "Execution Time (milliseconds)"]
table_data = [[size, a_val, b_val, "{:.4f}".format(time)] for (size, a_val, b_val, time) in all_results_np]  # Format time to 4 decimal places
table_str = tabulate(table_data, headers=headers)

# Inserting the table into the scrolled text widget
text_area.insert(tk.END, table_str)

# Create the plot
fig, ax = plt.subplots()

for A_value in [1, 2, 0]:
    for B_value in [1, 2, 0]:
        data = [(size, time) for (size, a_val, b_val, time) in all_results_np if a_val == A_value and b_val == B_value]
        sizes, times = zip(*data)
        ax.plot(sizes, times, marker='o', label=f"A={A_value}, B={B_value}")

ax.set_xlabel('Matrix Size')
ax.set_ylabel('Execution Time (milliseconds)')
ax.set_title('Matrix Multiplication Benchmark (NumPy)')
ax.legend()

# Display the plot on the canvas
plt.tight_layout()
plt.close(fig)
canvas = FigureCanvasTkAgg(fig, master=canvas)
canvas.draw()
canvas.get_tk_widget().pack(expand=True, fill='both')

# Running the tkinter event loop
root.mainloop()
