using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers.Misc;
using NeuralNetworkNET.Networks.Activations.Delegates;

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
        public static float[,] Sum([NotNull] this float[,] m, [NotNull] float[] v)
        {
            // Execute the transposition in parallel
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            float[,] result = new float[h, w];
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pr = result, pm = m, pv = v)
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
        /// Sums two matrices element-wise
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        /// <param name="mode">Indicates the mode to use to sum the matrices</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Sum([NotNull] this float[,] m1, [NotNull] float[,] m2, MatrixSumMode mode)
        {
            // Execute the transposition in parallel
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1),
                m2w = m2.GetLength(1);
            if (m2.GetLength(0) != h) throw new ArgumentException(nameof(m2), "Invalid matrix size");
            float[,] result = new float[h, w];

            // Elementwise kernel
            void ElementwiseSum(int i)
            {
                unsafe
                {
                    fixed (float* pr = result, pm1 = m1, pm2 = m2)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int target = offset + j;
                            pr[target] = pm1[target] + pm2[target];
                        }
                    }
                }
            }

            // Column by column kernel
            void ColumnByColumnSum(int i)
            {
                unsafe
                {
                    fixed (float* pr = result, pm1 = m1, pm2 = m2)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int target = offset + j;
                            pr[target] = pm1[target] + pm2[i];
                        }
                    }
                }
            }

            // Select and execute the right sum mode
            ParallelLoopResult loopResult;
            switch (mode)
            {
                case MatrixSumMode.Elementwise:
                    if (m2w != w) throw new ArgumentException(nameof(m2), "The second matrix must have the same size as the first");
                    loopResult = Parallel.For(0, h, ElementwiseSum);
                    break;
                case MatrixSumMode.ColumnByColumn:
                    if (m2w != 1) throw new ArgumentException(nameof(m2), "The second matrix must be a column vector");
                    loopResult = Parallel.For(0, h, ColumnByColumnSum);
                    break;
                default:
                    throw new ArgumentOutOfRangeException("Unsupported sum mode");
            }
            if (!loopResult.IsCompleted) throw new Exception("Error while runnig the parallel loop");
            return result;
        }

        /// <summary>
        /// Sums a row vector to all the rows in the input matrix, with side effect
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="v">The vector to sum</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void InPlaceSum([NotNull] this float[,] m, [NotNull] float[] v)
        {
            // Execute the transposition in parallel
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pm = m, pv = v)
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
        [CollectionAccess(CollectionAccessType.Read)]
        public static void Subtract([NotNull] this float[,] m1, [NotNull] float[,] m2)
        {
            // Execute the transposition in parallel
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pm1 = m1, pm2 = m2)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int position = offset + j;
                            pm1[position] -= pm2[position];
                        }
                    }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
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
        public static float[] Multiply([NotNull] this float[] v, [NotNull] float[,] m)
        {
            // Check
            if (v.Length != m.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid inputs sizes");

            // Initialize the parameters and the result vector
            int w = m.GetLength(1);
            float[] result = new float[w];

            // Loop in parallel
            bool loopResult = Parallel.For(0, w, j =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each column
                    fixed (float* pm = result, p1 = v, p2 = m)
                    {
                        // Perform the multiplication
                        int j2 = j;
                        float res = 0;
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
        /// Performs the in place Hadamard product between two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void InPlaceHadamardProduct([NotNull] this float[,] m1, [NotNull] float[,] m2)
        {
            // Check
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");

            // Loop in parallel
            unsafe void Kernel(int i)
            {
                // Get the pointers and iterate fo each column
                fixed (float* pm1 = m1, pm2 = m2)
                {
                    // Perform the product
                    int offset = i * w;
                    for (int j = 0; j < w; j++)
                    {
                        int position = offset + j;
                        pm1[position] *= pm2[position];
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the Hadamard product between two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] HadamardProduct([NotNull] this float[,] m1, [NotNull] float[,] m2)
        {
            // Check
            int
                h = m1.GetLength(0),
                w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            float[,] result = new float[h, w];

            // Loop in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each column
                    fixed (float* pm = result, pm1 = m1, pm2 = m2)
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
        public static float[,] Multiply([NotNull] this float[,] m1, [NotNull] float[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            float[,] result = new float[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (float* pm = result, pm1 = m1, pm2 = m2)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            float res = 0;
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
        public static float[] Activation([NotNull] this float[] v, [NotNull] ActivationFunction activation)
        {
            float[] result = new float[v.Length];
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
        public static float[,] Activation([NotNull] this float[,] m, [NotNull] ActivationFunction activation)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            float[,] result = new float[h, w];

            // Execute the activation in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pr = result, pm = m)
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
        public static void InPlaceActivation([NotNull] this float[,] m, [NotNull] ActivationFunction activation)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);

            // Execute the activation in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pm = m)
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
        /// Performs the softmax normalization on the input matrix, dividing every value by the sum of all the values
        /// </summary>
        /// <param name="m">The matrix to normalize</param>
        public static void InPlaceSoftmaxNormalization([NotNull] this float[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            float[] partials = new float[h];

            // Partial sum
            unsafe void PartialSum(int i)
            {
                int offset = i * w;
                fixed (float* pp = partials, pm = m)
                {
                    float sum = 0;
                    for (int j = 0; j < w; j++)
                        sum += pm[offset + j];
                    pp[i] = sum;
                }
            }
            Parallel.For(0, h, PartialSum).AssertCompleted();

            // Normalization of the matrix values
            unsafe void NormalizationKernel(int i)
            {
                int offset = i * w;
                fixed (float* p = m, pp = partials)
                    for (int j = 0; j < w; j++)
                        p[offset + j] /= pp[i];
            }
            Parallel.For(0, h, NormalizationKernel);
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
        public static float[,] MultiplyAndActivation([NotNull] this float[,] m1, [NotNull] float[,] m2, [NotNull] ActivationFunction activation)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            float[,] result = new float[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (float* pm = result, pm1 = m1, pm2 = m2)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            float res = 0;
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
        public static float[,] MultiplyWithSum([NotNull] this float[,] m1, [NotNull] float[,] m2, [NotNull] float[] v)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            float[,] result = new float[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (float* pm = result, pm1 = m1, pm2 = m2, pv = v)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            float res = 0;
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
        public static float[,] MultiplyWithSumAndActivation([NotNull] this float[,] m1, [NotNull] float[,] m2, [NotNull] float[] v, [NotNull] ActivationFunction activation)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            float[,] result = new float[h, w];

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Get the pointers and iterate fo each row
                    fixed (float* pm = result, pm1 = m1, pm2 = m2, pv = v)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            float res = 0;
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
        /// Calculates d(l) by applying the Hadamard product of d(l + 1) and W(l)T and the activation prime of z
        /// </summary>
        /// <param name="z">The activity on the previous layer</param>
        /// <param name="delta">The first matrix to multiply</param>
        /// <param name="wt">The second matrix to multiply</param>
        /// <param name="prime">The activation prime function to use</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void InPlaceMultiplyAndHadamardProductWithAcrivationPrime(
            [NotNull] this float[,] z, [NotNull] float[,] delta, [NotNull] float[,] wt, [NotNull] ActivationFunction prime)
        {
            // Initialize the parameters and the result matrix
            int h = delta.GetLength(0);
            int w = wt.GetLength(1);
            int l = delta.GetLength(1);

            // Checks
            if (l != wt.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (h != z.GetLength(0) || w != z.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Execute the multiplication in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pz = z, pm1 = delta, pm2 = wt)
                    {
                        // Save the index and iterate for each column
                        int i1 = i * l;
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the multiplication
                            int i2 = j;
                            float res = 0;
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
        public static float[,] Transpose([NotNull] this float[,] m)
        {
            // Setup
            int h = m.GetLength(0), w = m.GetLength(1);
            float[,] result = new float[w, h];

            // Execute the transposition in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (float* pr = result, pm = m)
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
        public static (int x, int y, float value) Max([NotNull] this float[,] m)
        {
            // Checks and local variables setup
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input matrix can't be empty");
            if (m.Length == 1) return (0, 0, m[0, 0]);
            int
                h = m.GetLength(0),
                w = m.GetLength(1),
                x = 0, y = 0;
            float max = float.MinValue;

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
        public static float[,] Normalize([NotNull] this float[,] m)
        {
            // Setup
            if (m.Length == 0) return new float[0, 0];
            int h = m.GetLength(0), w = m.GetLength(1);
            (_, _, float max) = m.Max();
            float[,] normalized = new float[h, w];

            // Populate the normalized matrix
            bool result = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Fix the pointers and iterate on the current row
                    fixed (float* pn = normalized, pm = m)
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
        public static float[,] ToMatrix([NotNull] this float[] v)
        {
            // Preliminary checks and declarations
            if (v.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int length = v.Length;
            float[,] result = new float[1, length];

            // Copy the content
            Buffer.BlockCopy(v, 0, result, 0, sizeof(float) * length);
            return result;
        }

        /// <summary>
        /// Flattens the input matrix into a linear array
        /// </summary>
        /// <param name="m">The matrix to flatten</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[] Flatten([NotNull] this float[,] m)
        {
            // Preliminary checks and declarations
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int length = m.Length;
            float[] result = new float[length];

            // Copy the content
            Buffer.BlockCopy(m, 0, result, 0, sizeof(float) * length);
            return result;
        }

        /// <summary>
        /// Flattens the input volume in a linear array
        /// </summary>
        /// <param name="volume">The volume to flatten</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[] Flatten([NotNull, ItemNotNull] this IReadOnlyList<float[,]> volume)
        {
            // Preliminary checks and declarations
            if (volume.Count == 0) throw new ArgumentOutOfRangeException("The input volume can't be empty");
            int
                depth = volume.Count,
                length = volume[0].Length,
                bytes = sizeof(float) * length;
            float[] result = new float[depth * length];

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
        /// Splits the input matrix into two matrices with the desired number of rows each
        /// </summary>
        /// <param name="m">The source matrix to read from</param>
        /// <param name="rows">The number of rows in the first returned matrix</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        private static (float[,], float[,]) SplitRows([NotNull] float[,] m, int rows)
        {
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            if (rows >= h) throw new ArgumentOutOfRangeException(nameof(rows), "The number of rows must be smaller than the original height");
                float[,]
                m1 = new float[rows, w],
                m2 = new float[h - rows, w];
            Buffer.BlockCopy(m, 0, m1, 0, sizeof(float) * m1.Length);
            Buffer.BlockCopy(m, sizeof(float) * w * rows, m2, 0, sizeof(float) * m2.Length);
            return (m1, m2);
        }

        /// <summary>
        /// Merges the rows of the input matrices into a single matrix
        /// </summary>
        /// <param name="blocks">The matrices to merge</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] MergeRows([NotNull, ItemNotNull] this IReadOnlyList<float[,]> blocks)
        {
            // Preliminary checks and declarations
            if (blocks.Count == 0) throw new ArgumentOutOfRangeException("The blocks list can't be empty");
            int
                h = blocks.Sum(b => b.GetLength(0)),
                w = blocks[0].GetLength(1),
                rowBytes = sizeof(float) * w;
            float[,] result = new float[h, w];

            // Execute the copy in parallel
            int position = 0;
            for (int i = 0; i < blocks.Count; i++)
            {
                float[,] next = blocks[i];
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
        public static float[] CompressVertically([NotNull] this float[,] m)
        {
            // Preliminary checks and declarations
            if (m.Length == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            float[] vector = new float[w];

            // Compress the matrix
            Parallel.For(0, w, j =>
            {
                unsafe
                {
                    // Fix the pointers and add the current values
                    fixed (float* pm = m, pv = vector)
                    {
                        float sum = 0;
                        for (int i = 0; i < h; i++)
                            sum += pm[i * w + j];
                        pv[j] = sum;
                    }
                }
            }).AssertCompleted();
            return vector;
        }

        /// <summary>
        /// Extracts a series of serialized matrices from a single matrix
        /// </summary>
        /// <param name="m">The source matrix</param>
        [PublicAPI]
        [Pure, NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[][,] Extract3DVolume([NotNull] this float[,] m)
        {
            int
                h = m.GetLength(0),
                w = m.GetLength(1),
                axis = w.IntegerSquare();
            if (axis * axis != w) throw new ArgumentOutOfRangeException("Invalid matrix size");
            float[][,] raw = new float[h][,];
            int bytesize = sizeof(float) * w;
            Parallel.For(0, h, i =>
            {
                float[,] _2d = new float[axis, axis];
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
        public static void Tweak([NotNull] this float[,] m, Func<float, float> f)
        {
            int w = m.GetLength(1);
            bool result = Parallel.For(0, m.GetLength(0), i =>
            {
                unsafe
                {
                    fixed (float* p = m)
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

        #region Argmax

        /// <summary>
        /// Returns the index of the maximum value in the input vector
        /// </summary>
        /// <param name="p">A pointer to the buffer to read</param>
        /// <param name="length">The length of the buffer to consider</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax(float* p, int length)
        {
            if (length < 2) return 0;
            int index = 0;
            float max = float.MinValue;
            for (int j = 0; j < length; j++)
            {
                if (p[j] > max)
                {
                    max = p[j];
                    index = j;
                }
            }
            return index;
        }

        /// <summary>
        /// Returns the index of the maximum value in the input vector
        /// </summary>
        /// <param name="v">The input vector to read from</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static int Argmax([NotNull] this float[] v)
        {
            if (v.Length < 2) return 0;
            int index = 0;
            float max = float.MinValue;
            for (int j = 0; j < v.Length; j++)
            {
                if (v[j] > max)
                {
                    max = v[j];
                    index = j;
                }
            }
            return index;
        }

        /// <summary>
        /// Returns the index of the maximum value in the input matrix
        /// </summary>
        /// <param name="m">The input matrix to read from</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static int Argmax([NotNull] this float[,] m)
        {
            if (m.Length < 2) return 0;
            int index = 0;
            float max = float.MinValue;
            unsafe
            {
                fixed (float* p = m)
                    for (int i = 0; i < m.Length; i++)
                        if (p[i] > max)
                        {
                            max = p[i];
                            index = i;
                        }
            }
            return index;
        }

        #endregion

        #region BlockCopy

        /// <summary>
        /// Returns a deep copy of the input matrix
        /// </summary>
        /// <param name="m">The matrix to clone</param>
        /// <remarks>This method avoids the boxing of the <see cref="Array.Clone"/> method, and it is faster thanks to <see cref="Buffer.BlockCopy"/></remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] BlockCopy([NotNull] this float[,] m)
        {
            float[,] result = new float[m.GetLength(0), m.GetLength(1)];
            Buffer.BlockCopy(m, 0, result, 0, m.Length);
            return result;
        }

        /// <summary>
        /// Returns a deep copy of the input vector
        /// </summary>
        /// <param name="v">The vector to clone</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[] BlockCopy([NotNull] this float[] v)
        {
            float[] result = new float[v.Length];
            Buffer.BlockCopy(v, 0, result, 0, v.Length);
            return result;
        }

        #endregion

        #region Content check

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        public static bool ContentEquals([CanBeNull] this float[,] m, [CanBeNull] float[,] o)
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
        public static bool ContentEquals([CanBeNull] this float[] v, [CanBeNull] float[] o)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (!v[i].EqualsWithDelta(o[i])) return false;
            return true;
        }

        #endregion

        #region Debug

        /// <summary>
        /// Returns a formatted representation of the input matrix
        /// </summary>
        /// <param name="m">The matrix to convert to <see cref="String"/></param>
        [PublicAPI]
        [Pure, NotNull]
        public static String ToFormattedString([NotNull] this float[,] m)
        {
            if (m.Length == 0) return "{ { } }";
            int h = m.GetLength(0), w = m.GetLength(1);
            StringBuilder builder = new StringBuilder();
            builder.Append("{ ");
            for (int i = 0; i < h; i++)
            {
                if (w > 0)
                {
                    builder.Append("{ ");
                    for (int j = 0; j < w; j++)
                    {
                        builder.Append($"{m[i, j]}");
                        if (j < w - 1) builder.Append(", ");
                    }
                    builder.Append(" }");
                }
                builder.Append(i < h - 1 ? ",\n  " : " }");
            }
            return builder.ToString();
        }

        #endregion
    }
}
