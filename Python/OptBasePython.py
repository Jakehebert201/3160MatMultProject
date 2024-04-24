import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tabulate import tabulate
import tkinter as tk
from tkinter import scrolledtext

def matrix_multiply(A, B, size):
    C = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            c_ij = 0
            for k in range(size):
                c_ij += A[i][k] * B[k][j]
            C[i][j] = c_ij
    return C

def benchmark_matrix_multiplication(size):
    A_values = [1, 2, 0]  # All 1s, all 2s, and incrementing by 1
    B_values = [
        [[1 if i == j else 0 for i in range(size)] for j in range(size)],  # Identity matrix
        [[1 for _ in range(size)] for _ in range(size)],  # All 1s
        [[2 for _ in range(size)] for _ in range(size)]   # All 2s
    ]

    results = []

    for A_value in A_values:
        for B_value in B_values:
            start_time = time.time()
            C = matrix_multiply([[A_value] * size] * size, B_value, size)
            end_time = time.time()
            execution_time = end_time - start_time
            results.append((size, A_value, B_value[0][0], execution_time))

    return results

# Benchmarking matrix multiplication for sizes 128, 256, and 512
sizes = [128, 256, 512]
all_results = []

for size in sizes:
    all_results.extend(benchmark_matrix_multiplication(size))

# Constructing the table
headers = ["Size", "A Value", "B Value", "Execution Time (milliseconds)"]
table_data = [[size, a_val, b_val, "{:.4f}".format(time * 1000)] for (size, a_val, b_val, time) in all_results]  # Convert seconds to milliseconds
table_str = tabulate(table_data, headers=headers)

# Displaying results in a single window
root = tk.Tk()
root.title("Matrix Multiplication Benchmark Results")

# Create a frame to hold both the plot and the table
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame for the table
table_frame = tk.Frame(main_frame)
table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a scrolled text widget to display the table
text_area = scrolledtext.ScrolledText(table_frame, width=80, height=30)
text_area.pack(expand=True, fill='both')
text_area.insert(tk.END, table_str)

# Running the tkinter event loop
root.mainloop()
