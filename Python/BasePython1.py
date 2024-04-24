import time
import csv

def matrix_multiply(A, B, size):
    C = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
    return C

def generate_matrix(size, value):
    return [[value for _ in range(size)] for _ in range(size)]

def generate_multiplier_matrix(size, value):
    return [[value if i == j else 0 for i in range(size)] for j in range(size)]

def benchmark_matrix_multiplication(size):
    A_value = 1
    B_value = generate_multiplier_matrix(size, 1)  # B = 1

    results = []

    A = generate_matrix(size, A_value)
    B = B_value
    start_time = time.time()
    C = matrix_multiply(A, B, size)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    results.append(("INTERPRETER_UNOPT", size, A_value, B_value[0][0], execution_time))

    return results

# Benchmarking matrix multiplication for sizes 128, 256, 512, 1024, and 2048
sizes = [128, 256, 512, 1024, 2048]
all_results_interpreter = []

for size in sizes:
    all_results_interpreter.extend(benchmark_matrix_multiplication(size))

# Outputting results to a CSV file for interpreter version
interpreter_csv_file = "interpreter_results.csv"
with open(interpreter_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["STRING_LABEL", "Size", "A Value", "B Value", "Execution Time (milliseconds)"])
    for row in all_results_interpreter:
        writer.writerow(row)
