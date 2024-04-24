fun main(args: Array<String>) {
        // Define matrix sizes for benchmarking
        val matrixSizes = intArrayOf(128, 256, 512)
//        val warmupSizes = intArrayOf(1, 8, 16, 64, 128, 512, 1024)
        val warmupSizes = intArrayOf(100)

        for (warmup in warmupSizes) {

            // Perform matrix multiplication for each size
            for (size in matrixSizes) {
                // Create multiplicand matrix A and multiplier matrix B
                val A = generateMatrix(size)
                val B = generateIdentityMatrix(size)
                // Warm-up phase to force JIT optimization
                warmUp(A, B, size, warmup)

                // Perform matrix multiplication
                val startTime = System.nanoTime()
                var result = naiveMatrixMultiply(A, B, size)
                val endTime = System.nanoTime()

                // Calculate execution time in milliseconds
                var executionTime = (endTime - startTime) / 1e6

                // Print the result matrix and execution time
//            printMatrix(result)
                println("Kotlin Naive,$size,1,1,$executionTime")
                // Perform matrix multiplication
                result = optimizedNaiveMatrixMultiply(A, B, size)
                val startTimeOptimized = System.nanoTime()
                result = optimizedNaiveMatrixMultiply(A, B, size)
                val endTimeOptimized = System.nanoTime()

                // Calculate execution time in milliseconds
                executionTime = (endTimeOptimized - startTimeOptimized) / 1e6

                // Print the result matrix and execution time
//            printMatrix(result)
                println("Kotlin Optimized With $warmup Warmup's,$size,1,1,$executionTime")
//                System.out.println("Execution time for optimized matrix size " + size + ": " + executionTime + " milliseconds, "+warmup);
            }
        }
    }

    // Warm-up phase to force JIT optimization
    // Force the JIT to give our optimization a cheeky edge.
    fun warmUp(A: Array<IntArray>, B: Array<IntArray>, size: Int, iterations: Int) {
        // Adjust the number of iterations as needed
        repeat(iterations) {
            optimizedNaiveMatrixMultiply(A, B, size)
//            naiveMatrixMultiply(A, B, size)
        }
    }

    // Optimized naive matrix multiplication algorithm using loop unrolling
    fun optimizedNaiveMatrixMultiply(A: Array<IntArray>, B: Array<IntArray>, size: Int): Array<IntArray> {
        val C = Array(size) { IntArray(size) }
        for (i in 0 until size) {
            for (j in 0 until size) {
                var sum = 0
                for (k in 0 until size step 4) {
                    sum += A[i][k] * B[k][j]
                    sum += A[i][k + 1] * B[k + 1][j]
                    sum += A[i][k + 2] * B[k + 2][j]
                    sum += A[i][k + 3] * B[k + 3][j]
                }
                C[i][j] = sum
            }
        }
        return C
    }

    // Naive matrix multiplication algorithm
    fun naiveMatrixMultiply(A: Array<IntArray>, B: Array<IntArray>, size: Int): Array<IntArray> {
        val C = Array(size) { IntArray(size) }
        for (i in 0 until size) {
            for (j in 0 until size) {
                for (k in 0 until size) {
                    C[i][j] += A[i][k] * B[k][j]
                }
            }
        }
        return C
    }

    // Generate a matrix with elements of all 1s, all 2s, and incrementing by 1 per element
    fun generateMatrix(size: Int): Array<IntArray> {
        val matrix = Array(size) { IntArray(size) }
        var value = 1
        for (i in 0 until size) {
            for (j in 0 until size) {
                matrix[i][j] = value
                value++
            }
        }
        return matrix
    }

    // Generate an identity matrix
    fun generateIdentityMatrix(size: Int): Array<IntArray> {
        val matrix = Array(size) { IntArray(size) }
        for (i in 0 until size) {
            matrix[i][i] = 1
        }
        return matrix
    }

    // Print a matrix
    fun printMatrix(matrix: Array<IntArray>) {
        for (row in matrix) {
            for (value in row) {
                print("$value ")
            }
            println()
        }
    }