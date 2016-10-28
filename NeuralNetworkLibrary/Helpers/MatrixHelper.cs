using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkLibrary.Helpers
{
    /// <summary>
    /// A helper class with some methods to work with matrices
    /// </summary>
    public static class MatrixHelper
    {
        /// <summary>
        /// Iterate over all the positions in a given matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="action">The action to perform for each position</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ForEach(this double[,] m, Action<int, int> action)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    action(i, j);
                }
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        public static double[,] Multiply(double[,] m1, double[,] m2)
        {
            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,] result = new double[h, w];
            unsafe
            {
                // Get the pointers and iterate fo each row
                fixed (double* pm = result, pm1 = m1, pm2 = m2)
                {
                    for (int i = 0; i < h; i++)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            double res = 0;
                            for (int k = 0; k < l; k++, i2 += w)
                            {
                                res += pm1[i1 + k] * pm2[i2];
                            }
                            pm[i * w + j] = res;
                        }
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="z">The input to process</param>
        /// <param name="threshold">If not null, sets to 0 every input under that value</param>
        public static void Sigmoid(double[,] z, double? threshold = null)
        {
            // Iterate over all the matrix elements
            z.ForEach((i, j) =>
            {
                double z0 = 1 / (1 + Math.Exp(-z[i, j]));
                z[i, j] = threshold == null ? z0 : (z0 > threshold.Value ? z0 : 0);
            });
        }

        /// <summary>
        /// Performs a random two points crossover between two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        /// <param name="random">The random instance</param>
        public static double[,] TwoPointsCrossover(double[,] m1, double[,] m2, Random random)
        {
            // Get the size of the matrix and check the input
            int h = m1.GetLength(0), w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentOutOfRangeException();

            // Get the crossover range and iterate over the two matrices
            double[,] result = new double[h, w];
            Range xr = random.NextRange(h), yr = random.NextRange(w);
            m1.ForEach((i, j) =>
            {
                // Perform the crossover when needed
                if (i >= xr.Start && i <= xr.End && j >= yr.Start && j <= yr.End)
                {
                    result[i, j] = m2[i, j];
                }
                else result[i, j] = m1[i, j];
            });
            return result;
        }

        /// <summary>
        /// Performs random mutations on the target matrix
        /// </summary>
        /// <param name="m">The matrix to edit</param>
        /// <param name="threshold">The mutation threshold</param>
        /// <param name="r">The random instance</param>
        public static void RandomMutate(double[,] m, int threshold, Random r)
        {
            m.ForEach((i, j) =>
            {
                // Check if the mutation is necessary
                if (r.Next(100) >= threshold) return;

                // Mutate the weight
                if (r.NextBool())
                {
                    double diff = 1 - m[i, j];
                    if (r.NextBool()) m[i, j] += r.NextDouble() * diff;
                    else m[i, j] -= r.NextDouble() * diff;
                }
                else m[i, j] = r.NextGaussian();
            });
        }

        /// <summary>
        /// Returns a new matrix filled with random values from a random Gaussian distribution (0, 1)
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="random">The random instance</param>
        public static double[,] RandomMatrix(int x, int y, Random random)
        {
            double[,] result = new double[x, y];
            result.ForEach((i, j) => result[i, j] = random.NextGaussian());
            return result;
        }
    }
}
