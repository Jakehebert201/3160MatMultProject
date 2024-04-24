

public class main {
    public static void main(String[] args) {
        // Define matrix sizes for benchmarking
        int[] matrixSizes = {128, 256, 512};
//        int[] warmupSizes = {1,8,16,64,128,512,1024};
        int[] warmupSizes = {1000};



        for (int warmup : warmupSizes) {

            // Perform matrix multiplication for each size
            for (int size : matrixSizes) {
                // Create multiplicand matrix A and multiplier matrix B
                int[][] A = generateMatrix(size);
                int[][] B = generateIdentityMatrix(size);
                // Warm-up phase to force JIT optimization
                warmUp(A, B, size, warmup);

                // Perform matrix multiplication
                long startTime = System.nanoTime();
                int[][] result = naiveMatrixMultiply(A, B, size);
                long endTime = System.nanoTime();

                // Calculate execution time in milliseconds
                double executionTime = (endTime - startTime) / 1e6;

                // Print the result matrix and execution time
//            printMatrix(result);
                System.out.println("Naive,"+size+","+1+","+1+","+executionTime);
                // Perform matrix multiplication
                startTime = System.nanoTime();
                result = optimizedNaiveMatrixMultiply(A, B, size);
                endTime = System.nanoTime();

                // Calculate execution time in milliseconds
                executionTime = (endTime - startTime) / 1e6;

                // Print the result matrix and execution time
//            printMatrix(result);
                System.out.println("Optimized With "+warmup+" Warmup's"+","+size+","+1+","+1+","+executionTime);
//                System.out.println("Execution time for optimized matrix size " + size + ": " + executionTime + " milliseconds, "+warmup);


            }
        }
    }
    // Warm-up phase to force JIT optimization
    // Force the JIT to give our optimization a cheeky edge.
    public static void warmUp(int[][] A, int[][] B, int size,int iterations) {
         // Adjust the number of iterations as needed
        for (int i = 0; i < iterations; i++) {
            optimizedNaiveMatrixMultiply(A, B, size);
//            naiveMatrixMultiply(A,B,size);
        }
    }

    // Optimized naive matrix multiplication algorithm using loop unrolling
    public static int[][] optimizedNaiveMatrixMultiply(int[][] A, int[][] B, int size) {
        int[][] C = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int sum = 0;
                for (int k = 0; k < size; k += 4) {
                    sum += A[i][k] * B[k][j];
                    sum += A[i][k + 1] * B[k + 1][j];
                    sum += A[i][k + 2] * B[k + 2][j];
                    sum += A[i][k + 3] * B[k + 3][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    // Naive matrix multiplication algorithm
    public static int[][] naiveMatrixMultiply(int[][] A, int[][] B, int size) {
        int[][] C = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // Generate a matrix with elements of all 1s, all 2s, and incrementing by 1 per element
    public static int[][] generateMatrix(int size) {
        int[][] matrix = new int[size][size];
        int value = 1;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i][j] = value;
                value++;
            }
        }
        return matrix;
    }

    // Generate an identity matrix
    public static int[][] generateIdentityMatrix(int size) {
        int[][] matrix = new int[size][size];
        for (int i = 0; i < size; i++) {
            matrix[i][i] = 1;
        }
        return matrix;
    }

    // Print a matrix
    public static void printMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            for (int value : row) {
                System.out.print(value + " ");
            }
            System.out.println();
        }
    }
}
