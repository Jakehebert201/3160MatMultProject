import matplotlib.pyplot as plt
from tabulate import tabulate
import tkinter as tk

# Define the data
new_data_1 = [
    ("Naive", 128, 1, 1, 32.9671),
    ("Optimized With 1 Warmup's", 128, 1, 1, 6.632),
    ("Naive", 256, 1, 1, 71.9148),
    ("Optimized With 1 Warmup's", 256, 1, 1, 57.2151),
    ("Naive", 512, 1, 1, 650.4518),
    ("Optimized With 1 Warmup's", 512, 1, 1, 622.1013),
]

new_data_2 = [
    ("Naive", 128, 1, 1, 33.7377),
    ("Optimized With 10 Warmup's", 128, 1, 1, 3.8319),
    ("Naive", 256, 1, 1, 83.183),
    ("Optimized With 10 Warmup's", 256, 1, 1, 64.1178),
    ("Naive", 512, 1, 1, 500.0063),
    ("Optimized With 10 Warmup's", 512, 1, 1, 602.7179),
]

new_data_3 = [
    ("Naive", 128, 1, 1, 34.5482),
    ("Optimized With 100 Warmup's", 128, 1, 1, 7.8845),
    ("Naive", 256, 1, 1, 60.7748),
    ("Optimized With 100 Warmup's", 256, 1, 1, 27.0278),
    ("Naive", 512, 1, 1, 356.6399),
    ("Optimized With 100 Warmup's", 512, 1, 1, 327.9183),
    ("Naive", 128, 1, 1, 23.7465),
    ("Optimized With 1000 Warmup's", 128, 1, 1, 2.8633),
    ("Naive", 512, 1, 1, 323.6856),
    ("Optimized With 1000 Warmup's", 512, 1, 1, 303.0873),
]

# Prepare data for plotting
data_sets = [new_data_1, new_data_2, new_data_3]
labels = ['New Data 1', 'New Data 2', 'New Data 3']
colors = ['blue', 'green', 'orange']
markers = ['o', 's', '^']

# Plotting
plt.figure(figsize=(10, 6))

for i, data in enumerate(data_sets):
    data_sorted = sorted(data, key=lambda x: x[4])  # Sort by execution time
    sizes = [item[1] for item in data_sorted]
    times = [item[4] for item in data_sorted]
    plt.plot(sizes, times, linestyle='-', marker=markers[i], color=colors[i], label=labels[i])

plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (milliseconds)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for the table
table_data = []
for data in data_sets:
    for item in data:
        table_data.append(item)

# Add a row of column headers
table_data.insert(0, ('Algorithm', 'Matrix Size', 'Warmups', 'Runs', 'Execution Time (milliseconds)'))

# Create a table using the tabulate library
table = tabulate(table_data, headers='firstrow', tablefmt='grid')

# Create a tkinter window to display the table
window = tk.Tk()
window.title('Execution Times Table')
window.geometry('600x400')

# Create a text widget to display the table
table_widget = tk.Text(window, width=100, height=20)
table_widget.pack(expand=True, fill='both')

# Insert the table into the text widget
table_widget.insert('1.0', table)

# Center the text in the table cells
table_widget.tag_configure('centertag', justify='center')
table_widget.tag_add('centertag', '1.0', 'end')

window.mainloop()