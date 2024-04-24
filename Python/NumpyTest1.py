import time
import numpy as np
import csv

def matrix_multiply_np(A, B):
    return np.dot(A, B)

def generate_matrix_np(size, value):
    return np.full((size, size), value)

def generate_multiplier_matrix_np(size, value):
    return np.eye(size) * value

def benchmark_matrix_multiplication_np(size):
    A_value = 1
    B_value = generate_multiplier_matrix_np(size, 1)  # B = 1

    results = []

    A = generate_matrix_np(size, A_value)
    B = B_value
    start_time = time.time()
    C = matrix_multiply_np(A, B)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    results.append(("NUMPY_UNOPT", size, A_value, B_value[0][0], execution_time))

    return results

# Benchmarking matrix multiplication for sizes 128, 256, 512, 1024, and 2048
sizes = [128, 256, 512, 1024, 2048]
all_results_numpy = []

for size in sizes:
    all_results_numpy.extend(benchmark_matrix_multiplication_np(size))

# Outputting results to a CSV file for numpy version
numpy_csv_file = "numpy_results.csv"
with open(numpy_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["STRING_LABEL", "Size", "A Value", "B Value", "Execution Time (milliseconds)"])
    for row in all_results_numpy:
        writer.writerow(row)
