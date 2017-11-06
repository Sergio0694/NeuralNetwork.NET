using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Architecture;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class MatrixExtensions
    {
        #region Sum

        /// <summary>
        /// Sums a row vector to all the rows in the input matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="v">The vector to sum</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Sum([NotNull] this double[,] m, [NotNull] double[] v)
        {
            // Execute the transposition in parallel
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            double[,] result = new double[h, w];
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m, pv = v)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                            pr[offset + j] = pm[offset + j] + pv[j];
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Sums a row vector to all the rows in the input matrix, with side effect
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="v">The vector to sum</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void InPlaceSum([NotNull] this double[,] m, [NotNull] double[] v)
        {
            // Execute the transposition in parallel
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pm = m, pv = v)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                            pm[offset + j] += pv[j];
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
        }

        /// <summary>
        /// Subtracts two matrices, element wise
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Subtract([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Execute the transposition in parallel
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            double[,] result = new double[h, w];
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm1 = m1, pm2 = m2)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int position = offset + j;
                            pr[position] = pm1[position] - pm2[position];
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Calculates half the sum of the squared difference of each value pair in the two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double HalfSquaredDifference([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Detect the size of the inputs
            int h = m1.GetLength(0), w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Calculate the cost (half the squared difference)
            double[] v = new double[h];
            bool result = Parallel.For(0, h, i =>
            {
                for (int j = 0; j < w; j++)
                {
                    double
                        delta = m1[i, j] - m2[i, j],
                        square = delta * delta;
                    v[i] += square;
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while runnig the parallel loop");

            // Sum the partial costs
            double cost = 0;
            for (int i = 0; i < h; i++) cost += v[i];
            return cost / 2;
        }

        #endregion

        #region Multiplication

        /// <summary>
        /// Performs the multiplication between a vector and a matrix
        /// </summary>
        /// <param name="v">The input vector</param>
        /// <param name="m">The matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Multiply([NotNull] this double[] v, [NotNull] double[,] m)
        {
            // Check
            if (v.Length != m.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid inputs sizes");

            // Initialize the parameters and the result vector
            int w = m.GetLength(1);
            double[] result = new double[w];

            // Loop in parallel
            bool loopResult = Parallel.For(0, w, j =>
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
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Performs the Hadamard product between two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] HadamardProduct([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Check
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            double[,] result = new double[h, w];

            // Loop in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each column
                    fixed (double* pm = result, pm1 = m1, pm2 = m2)
                    {
                        // Perform the product
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int position = offset + j;
                            pm[position] = pm1[position] * pm2[position];
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
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
            bool loopResult = Parallel.For(0, h, i =>
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
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        #endregion

        #region Activation

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        /// <param name="activation">The activation function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Activation([NotNull] this double[] v, [NotNull] Func<double, double> activation)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
                result[i] = activation(v[i]);
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="m">The input to process</param>
        /// <param name="activation">The activation function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Activation([NotNull] this double[,] m, [NotNull] Func<double, double> activation)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the activation in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                            pr[i * w + j] = activation(pm[i * w + j]);
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Applies the activation function to the input matrix
        /// </summary>
        /// <param name="m">The input to process</param>
        /// <param name="activation">The activation function to use</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void InPlaceActivation([NotNull] this double[,] m, [NotNull] Func<double, double> activation)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);

            // Execute the activation in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pm = m)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            int offset = i * w + j;
                            pm[offset] = activation(pm[offset]);
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
        }

        /// <summary>
        /// Returns the result of the input after the activation function primed has been applied
        /// </summary>
        /// <param name="v">The input to process</param>
        /// <param name="prime">The activation prime function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] ActivationPrime([NotNull] this double[] v, [NotNull] Func<double, double> prime)
        {
            double[] result = new double[v.Length];
            for (int i = 0; i < v.Length; i++)
            {
                result[i] = prime(v[i]);
            }
            return result;
        }

        /// <summary>
        /// Returns the result of the input after the activation function primed has been applied
        /// </summary>
        /// <param name="m">The input to process</param>
        /// <param name="prime">The activation pime function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] ActivationPrime([NotNull] this double[,] m, [NotNull] Func<double, double> prime)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the activation prime in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            pr[i * w + j] = prime(pm[i * w + j]);
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        #endregion

        #region Combined

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the activation function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="activation">The activation function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndActivation([NotNull] this double[,] m1, [NotNull] double[,] m2, [NotNull] Func<double, double> activation)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
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
                            pm[i * w + j] = activation(res); // Store the result and apply the activation
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices and sums another vector to the result
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyWithSum([NotNull] this double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (double* pm = result, pm1 = m1, pm2 = m2, pv = v)
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
                            pm[i * w + j] = res + pv[j]; // Sum the input vector to each component
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the activation function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix before applying the activation function</param>
        /// <param name="activation">The activation function to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyWithSumAndActivation([NotNull] this double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v, [NotNull] Func<double, double> activation)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,] result = new double[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (double* pm = result, pm1 = m1, pm2 = m2, pv = v)
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
                            res += pv[j]; // Sum the input vector to each component
                            pm[i * w + j] = activation(res); // Store the result and apply the activation
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Calculates d(L) by applying the Hadamard product of (yHat - y) and the activation prime of z
        /// </summary>
        /// <param name="a">The estimated y</param>
        /// <param name="y">The expected y</param>
        /// <param name="z">The activity on the last layer</param>
        /// <param name="prime">The activation prime function to use</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void InPlaceSubtractAndHadamardProductWithActivationPrime(
            [NotNull] this double[,] a, [NotNull] double[,] y, [NotNull] double[,] z, [NotNull] Func<double, double> prime)
        {
            // Checks
            int
                h = a.GetLength(0),
                w = a.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1) ||
                h != z.GetLength(0) || w != z.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (double* pa = a, py = y, pz = z)
                    {
                        // Save the index and iterate for each column
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int index = offset + j;
                            double
                                difference = pa[index] - py[index],
                                zPrime = prime(pz[index]),
                                hProduct = difference * zPrime;
                            pa[index] = hProduct;
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
        }

        /// <summary>
        /// Calculates d(l) by applying the Hadamard product of d(l + 1) and W(l)T and the activation prime of z
        /// </summary>
        /// <param name="z">The activity on the previous layer</param>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="prime">The activation prime function to use</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void MultiplyAndInPlaceActivationPrimeAndHadamardProduct(
            [NotNull] this double[,] z, [NotNull] double[,] m1, [NotNull] double[,] m2, [NotNull] Func<double, double> prime)
        {
            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);

            // Checks
            if (l != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (h != z.GetLength(0) || w != z.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pz = z, pm1 = m1, pm2 = m2)
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

                            // res has now the matrix multiplication result for position [i, j]
                            int zIndex = i * w + j;
                            pz[zIndex] = prime(pz[zIndex]) * res;
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
        }

        #endregion

        #region Misc

        /// <summary>
        /// Transposes the input matrix
        /// </summary>
        /// <param name="m">The matrix to transpose</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Transpose([NotNull] this double[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] result = new double[w, h];

            // Execute the transposition in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* pr = result, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                            pr[j * h + i] = pm[i * w + j];
                    }
                }
            }).IsCompleted;
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
            double max = Double.MinValue;

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
        /// Normalizes the values in a matrix in the [0..1] range
        /// </summary>
        /// <param name="m">The input matrix to normalize</param>
        [PublicAPI]
        [Pure, NotNull]
        public static double[,] Normalize([NotNull] this double[,] m)
        {
            // Setup
            if (m.Length == 0) return new double[0, 0];
            int h = m.GetLength(0), w = m.GetLength(1);
            (_, _, double max) = m.Max();
            double[,] normalized = new double[h, w];

            // Populate the normalized matrix
            bool result = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Fix the pointers and iterate on the current row
                    fixed (double* pn = normalized, pm = m)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            int index = i * w + j;
                            pn[index] = pm[index] / max;
                        }
                    }
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while runnig the parallel loop");
            return normalized;
        }

        /// <summary>
        /// Copies the input array into a matrix with a single row
        /// </summary>
        /// <param name="v">The array to copy</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] ToMatrix([NotNull] this double[] v)
        {
            // Preliminary checks and declarations
            if (v.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int length = v.Length;
            double[,] result = new double[1, length];

            // Copy the content
            Buffer.BlockCopy(v, 0, result, 0, sizeof(double) * length);
            return result;
        }

        /// <summary>
        /// Flattens the input matrix into a linear array
        /// </summary>
        /// <param name="m">The matrix to flatten</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Flatten([NotNull] this double[,] m)
        {
            // Preliminary checks and declarations
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int length = m.Length;
            double[] result = new double[length];

            // Copy the content
            Buffer.BlockCopy(m, 0, result, 0, sizeof(double) * length);
            return result;
        }

        /// <summary>
        /// Flattens the input volume in a linear array
        /// </summary>
        /// <param name="volume">The volume to flatten</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Flatten([NotNull, ItemNotNull] this IReadOnlyList<double[,]> volume)
        {
            // Preliminary checks and declarations
            if (volume.Count == 0) throw new ArgumentOutOfRangeException("The input volume can't be empty");
            int
                depth = volume.Count,
                length = volume[0].Length,
                bytes = sizeof(double) * length;
            double[] result = new double[depth * length];

            // Execute the copy in parallel
            bool loopResult = Parallel.For(0, depth, i =>
            {
                // Copy the volume data
                Buffer.BlockCopy(volume[i], 0, result, bytes * i, bytes);
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Merges the rows of the input matrices into a single matrix
        /// </summary>
        /// <param name="blocks">The matrices to merge</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MergeRows([NotNull, ItemNotNull] this IReadOnlyList<double[,]> blocks)
        {
            // Preliminary checks and declarations
            if (blocks.Count == 0) throw new ArgumentOutOfRangeException("The blocks list can't be empty");
            int
                h = blocks.Sum(b => b.GetLength(0)),
                w = blocks[0].GetLength(1),
                rowBytes = sizeof(double) * w;
            double[,] result = new double[h, w];

            // Execute the copy in parallel
            int position = 0;
            for (int i = 0; i < blocks.Count; i++)
            {
                double[,] next = blocks[i];
                if (next.GetLength(1) != w) throw new ArgumentOutOfRangeException("The blocks must all have the same width");
                int rows = next.GetLength(0);
                Buffer.BlockCopy(next, 0, result, rowBytes * position, rowBytes * rows);
                position += rows;
            }
            return result;
        }

        /// <summary>
        /// Compresses a matrix into a row vector by summing the components column by column
        /// </summary>
        /// <param name="m">The matrix to compress</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] CompressVertically([NotNull] this double[,] m)
        {
            // Preliminary checks and declarations
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            double[] vector = new double[w];

            // Compress the matrix
            bool result = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Fix the pointers and add the current values
                    fixed (double* pm = m, pv = vector)
                        for (int j = 0; j < w; j++)
                            pv[j] += pm[i * w + j];
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while runnig the parallel loop");
            return vector;
        }

        /// <summary>
        /// Extracts a series of serialized matrices from a single matrix
        /// </summary>
        /// <param name="m">The source matrix</param>
        [PublicAPI]
        [Pure, NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[][,] Extract3DVolume([NotNull] this double[,] m)
        {
            int
                h = m.GetLength(0),
                w = m.GetLength(1),
                axis = w.IntegerSquare();
            if (axis * axis != w) throw new ArgumentOutOfRangeException("Invalid matrix size");
            double[][,] raw = new double[h][,];
            int bytesize = sizeof(double) * w;
            Parallel.For(0, h, i =>
            {
                double[,] _2d = new double[axis, axis];
                Buffer.BlockCopy(m, i * bytesize, _2d, 0, bytesize);
                raw[i] = _2d;
            });
            return raw;
        }

        /// <summary>
        /// Edits the contents of the given matrix by applying a function to every item
        /// </summary>
        /// <param name="m">The matrix to edit</param>
        /// <param name="f">The function to modify the matrix elements</param>
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void Tweak([NotNull] this double[,] m, Func<double, double> f)
        {
            int w = m.GetLength(1);
            bool result = Parallel.For(0, m.GetLength(0), i =>
            {
                unsafe
                {
                    fixed (double* p = m)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int target = offset + j;
                            p[target] = f(p[target]);
                        }
                    }
                }
            }).IsCompleted;
            if (!result) throw new Exception("Error while runnig the parallel loop");
        }

        #endregion

        #region Content check

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        public static bool ContentEquals([CanBeNull] this double[,] m, [CanBeNull] double[,] o)
        {
            if (m == null && o == null) return true;
            if (m == null || o == null) return false;
            if (m.GetLength(0) != o.GetLength(0) ||
                m.GetLength(1) != o.GetLength(1)) return false;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (!m[i, j].EqualsWithDelta(o[i, j])) return false;
            return true;
        }

        /// <summary>
        /// Checks if two vectors have the same size and content
        /// </summary>
        /// <param name="v">The first vector to test</param>
        /// <param name="o">The second vector to test</param>
        public static bool ContentEquals([CanBeNull] this double[] v, [CanBeNull] double[] o)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (!v[i].EqualsWithDelta(o[i])) return false;
            return true;
        }

        #endregion
    }
}
