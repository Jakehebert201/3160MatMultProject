#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 ITERATIONS"
    exit 1
fi

# Run different samples of matrix multiplication and output to a file
ITERATIONS="$1"

echo "" >> CPUResults.txt

# Starts at 32 square matrix multiplication, ends at user-defined limit
for ((n=5; n <= ITERATIONS; n++))
do
    size=$((2**n))
    echo "CPU MAT MULT for size: $size" >> CPUResults.txt
    echo ""
    echo "Running matrix multiplication for size $size"
    start=$(date +%s%N) # Capture start time in nanoseconds
    ./copt 3 $size 1 | grep -Ev "^Running" >> CPUResults.txt
    end=$(date +%s%N) # Capture end time in nanoseconds
    elapsed=$(( ($end - $start) / 1000000 )) # Convert elapsed time to milliseconds
    echo "---------------------------------------" >> CPUResults.txt
    if (( $n < 10 )); then
        echo "Elapsed time for $size : $elapsed ms"
    else
        # For n >= 12, convert and display the time in seconds
        seconds=$(($elapsed / 1000))
        echo "Elapsed time for $size : $seconds s"
    fi
    echo "Elapsed time for $size displayed in results file."
done
