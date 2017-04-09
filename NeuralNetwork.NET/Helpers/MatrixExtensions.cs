using System;
using JetBrains.Annotations;

namespace NeuralNetwork.NET.Helpers
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class MatrixExtensions
    {
        /// <summary>
        /// Performs the multiplication between a vector and a matrix
        /// </summary>
        /// <param name="v">The input vector</param>
        /// <param name="m">The matrix to multiply</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Multiply([NotNull] this double[] v, [NotNull] double[,] m)
        {
            // Check
            if (v.Length != m.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid inputs sizes");

            // Initialize the parameters and the result vector
            int w = m.GetLength(1);
            double[] result = new double[w];

            // Loop in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, w, j =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each column
                    fixed (double* pm = result, p1 = v, p2 = m)
                    {
                        // Perform the multiplication
                        int j2 = j;
                        double res = 0;
                        for (int k = 0; k < v.Length; k++, j2 += w)
                        {
                            res += p1[k] * p2[j2];
                        }
                        pm[j] = res;
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Multiply([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the multiplication in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (double* pm = result, pm1 = m1, pm2 = m2)
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
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Transposes the input matrix
        /// </summary>
        /// <param name="m">The matrix to transpose</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Transpose([NotNull] this double[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[w, h];

            // Execute the transposition in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                            pr[j * h + i] = pm[i * w + j];
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Sigmoid([NotNull] this double[] v)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
                result[i] = 1 / (1 + Math.Exp(-v[i]));
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="m">The input to process</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Sigmoid([NotNull] this double[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the sigmoid in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                            pr[i * w + j] = 1 / (1 + Math.Exp(-pm[i * w + j]));
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function primed has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] SigmoidPrime([NotNull] this double[] v)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                double
                    exp = Math.Exp(-v[i]),
                    sum = 1 + exp,
                    square = sum * sum,
                    div = exp / square;
                result[i] = div;
            }
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function primed has been applied
        /// </summary>
        /// <param name="m">The input to process</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] SigmoidPrime([NotNull] this double[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the sigmoid prime in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            double
                                exp = Math.Exp(-pm[i * w + j]),
                                sum = 1 + exp,
                                square = sum * sum,
                                div = exp / square;
                            pr[i * w + j] = div;
                        }
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Calculates the position and the value of the biggest item in a matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        public static (int x, int y, double value) Max([NotNull] this double[,] m)
        {
            // Checks and local variables setup
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input matrix can't be empty");
            if (m.Length == 1) return (0, 0, m[0, 0]);
            int
                h = m.GetLength(0),
                w = m.GetLength(1),
                x = 0, y = 0;
            double max = double.MinValue;

            // Find the maximum value and its position
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    if (!(m[i, j] > max)) continue;
                    max = m[i, j];
                    x = i;
                    y = j;
                }
            return (x, y, max);
        }

        /// <summary>
        /// Flattens the input volume in a linear array
        /// </summary>
        /// <param name="volume">The volume to flatten</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Flatten([NotNull] this double[][,] volume)
        {
            // Preliminary checks and declarations
            if (volume.Length == 0) throw new ArgumentOutOfRangeException("The input volume can't be empty");
            int
                depth = volume.Length,
                h = volume[0].GetLength(0),
                w = volume[0].GetLength(1);
            double[] result = new double[h * w * depth];

            // Execute the copy in parallel
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, depth, i =>
            {
                // Copy the volume data
                unsafe
                {
                    fixed (double* r = result, p = volume[i])
                    {
                        // Copy each 2D matrix
                        for (int j = 0; j < h; j++)
                        for (int z = 0; z < w; z++)
                            r[h * w * i + j * w + z] = p[j * w + z];
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }
    }
}
