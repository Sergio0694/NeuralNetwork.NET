using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class MatrixExtensions
    {
        #region Subtraction

        /// <summary>
        /// Subtracts two matrices, element wise
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second</param>
        internal static unsafe void Subtract(in this Tensor m1, in Tensor m2)
        {
            int
                h = m1.Entities,
                w = m1.Length;
            if (h != m2.Entities || w != m2.Length) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");

            // Subtract in parallel
            float* pm1 = m1, pm2 = m2;
            void Kernel(int i)
            {
                int offset = i * w;
                for (int j = 0; j < w; j++)
                {
                    int position = offset + j;
                    pm1[position] -= pm2[position];
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        #endregion

        #region Multiplication

        /// <summary>
        /// Performs the in place Hadamard product between two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        internal static unsafe void InPlaceHadamardProduct(in this Tensor m1, in Tensor m2)
        {
            // Check
            int
                h = m1.Entities,
                w = m1.Length;
            if (h != m2.Entities || w != m2.Length) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            float* pm1 = m1, pm2 = m2;

            // Loop in parallel
            void Kernel(int i)
            {
                int offset = i * w;
                for (int j = 0; j < w; j++)
                {
                    int position = offset + j;
                    pm1[position] *= pm2[position];
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the in place Hadamard product between the activation of the first matrix and the second matrix
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        /// <param name="activation">The activation function to use</param>
        internal static unsafe void InPlaceActivationAndHadamardProduct(in this Tensor m1, in Tensor m2, [NotNull] ActivationFunction activation)
        {
            // Check
            int
                h = m1.Entities,
                w = m1.Length;
            if (h != m2.Entities || w != m2.Length) throw new ArgumentException(nameof(m2), "The two matrices must be of equal size");
            float* pm1 = m1, pm2 = m2;

            // Loop in parallel
            void Kernel(int i)
            {
                int offset = i * w;
                for (int j = 0; j < w; j++)
                {
                    int position = offset + j;
                    pm1[position] = activation(pm1[position]) * pm2[position];
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="result">The resulting matrix</param>
        internal static unsafe void Multiply(in this Tensor m1, in Tensor m2, out Tensor result)
        {
            // Initialize the parameters and the result matrix
            if (m1.Length != m2.Entities) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            int
                h = m1.Entities,
                w = m2.Length,
                l = m1.Length;
            Tensor.New(h, w, out result);
            float* pm = result, pm1 = m1, pm2 = m2;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
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
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        #endregion

        #region Activation

        /// <summary>
        /// Returns the result of the input after the activation function has been applied
        /// </summary>
        /// <param name="m">The input to process</param>
        /// <param name="activation">The activation function to use</param>
        /// <param name="result">The resulting matrix</param>
        internal static unsafe void Activation(in this Tensor m, [NotNull] ActivationFunction activation, out Tensor result)
        {
            // Setup
            int h = m.Entities, w = m.Length;
            Tensor.New(h, w, out result);
            float* pr = result, pm = m;

            // Execute the activation in parallel
            void Kernel(int i)
            {
                for (int j = 0; j < w; j++)
                    pr[i * w + j] = activation(pm[i * w + j]);
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the softmax normalization on the input matrix, dividing every value by the sum of all the values
        /// </summary>
        /// <param name="m">The matrix to normalize</param>
        internal static unsafe void InPlaceSoftmaxNormalization(in this Tensor m)
        {
            // Setup
            int h = m.Entities, w = m.Length;
            Tensor.New(1, h, out Tensor partials);
            float* pp = partials, pm = m;

            // Partial sum
            unsafe void PartialSum(int i)
            {
                int offset = i * w;
                float sum = 0;
                for (int j = 0; j < w; j++)
                    sum += pm[offset + j];
                pp[i] = sum;
            }
            Parallel.For(0, h, PartialSum).AssertCompleted();

            // Normalization of the matrix values
            unsafe void NormalizationKernel(int i)
            {
                int offset = i * w;
                for (int j = 0; j < w; j++)
                    pm[offset + j] /= pp[i];
            }
            Parallel.For(0, h, NormalizationKernel).AssertCompleted();
            partials.Free();
        }

        #endregion

        #region Combined

        /// <summary>
        /// Performs the multiplication between two matrices and sums another vector to the result
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix</param>
        /// <param name="result">The resulting matrix</param>
        internal static unsafe void MultiplyWithSum(in this Tensor m1, in Tensor m2, [NotNull] float[] v, out Tensor result)
        {
            // Initialize the parameters and the result matrix
            if (m1.Length != m2.Entities) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            int
                h = m1.Entities,
                w = m2.Length,
                l = m1.Length;
            Tensor.New(h, w, out result);
            float* pm = result, pm1 = m1, pm2 = m2;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
                fixed (float* pv = v)
                {
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
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Calculates d(l) by applying the Hadamard product of d(l + 1) and W(l)T and the activation prime of z
        /// </summary>
        /// <param name="z">The activity on the previous layer</param>
        /// <param name="delta">The first matrix to multiply</param>
        /// <param name="wt">The second matrix to multiply</param>
        /// <param name="prime">The activation prime function to use</param>
        internal static unsafe void InPlaceMultiplyAndHadamardProductWithActivationPrime(
            in this Tensor z, in Tensor delta, in Tensor wt, [NotNull] ActivationFunction prime)
        {
            // Initialize the parameters and the result matrix
            int h = delta.Entities;
            int w = wt.Length;
            int l = delta.Length;

            // Checks
            if (l != wt.Entities) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (h != z.Entities || w != z.Length) throw new ArgumentException("The matrices must be of equal size");
            float* pz = z, pm1 = delta, pm2 = wt;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
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
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        #endregion

        #region Misc

        /// <summary>
        /// Transposes the input matrix
        /// </summary>
        /// <param name="m">The matrix to transpose</param>
        /// <param name="result">The resulting matrix</param>
        internal static unsafe void Transpose(in this Tensor m, out Tensor result)
        {
            // Setup
            int h = m.Entities, w = m.Length;
            Tensor.New(w, h, out result);
            float* pr = result;

            // Execute the transposition in parallel
            float* pm = m;
            void Kernel(int i)
            {
                for (int j = 0; j < w; j++)
                    pr[j * h + i] = pm[i * w + j];
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Compresses a matrix into a row vector by summing the components column by column
        /// </summary>
        /// <param name="m">The matrix to compress</param>
        /// <param name="result">The resulting vector</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        internal static unsafe void CompressVertically(in this Tensor m, out Tensor result)
        {
            // Preliminary checks and declarations
            if (m.Entities == 0) throw new ArgumentOutOfRangeException("The input array can't be empty");
            int
                h = m.Entities,
                w = m.Length;
            Tensor.New(1, w, out result);
            float* pm = m, pv = result;

            // Compress the matrix
            void Kernel(int j)
            {
                float sum = 0;
                for (int i = 0; i < h; i++)
                    sum += pm[i * w + j];
                pv[j] = sum;
            }
            Parallel.For(0, w, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Calculates the position and the value of the biggest item in a matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        internal static (int x, int y, float value) Max([NotNull] this float[,] m)
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
        /// Finds the minimum and maximum value in the input memory area
        /// </summary>
        /// <param name="p">A pointer to the memory area to scan</param>
        /// <param name="length">The number of items to scan</param>
        internal static unsafe (float Min, float Max) MinMax(float* p, int length)
        {
            if (length == 0) return (0, 0);
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < length; i++)
            {
                float value = p[i];
                if (value < min) min = value;
                if (value > max) max = value;
            }
            return (min, max);
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
        /// Returns a matrix with the reshaped content of the input vector
        /// </summary>
        /// <param name="v">The input vector</param>
        /// <param name="h">The number of matrix rows</param>
        /// <param name="w">The number of matrix columns</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] AsMatrix([NotNull] this float[] v, int h, int w)
        {
            if (h * w != v.Length) throw new ArgumentException("The input dimensions don't match the source vector size");
            float[,] m = new float[h, w];
            Buffer.BlockCopy(v, 0, m, 0, sizeof(float) * v.Length);
            return m;
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
        /// Merges the input samples into a matrix dataset
        /// </summary>
        /// <param name="samples">The vectors to merge</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static (float[,], float[,]) MergeRows([NotNull] this IReadOnlyList<(float[] X, float[] Y)> samples)
        {
            // Preliminary checks and declarations
            if (samples.Count == 0) throw new ArgumentOutOfRangeException("The samples list can't be empty");
            int
                xLength = samples[0].X.Length,
                yLength = samples[0].Y.Length;
            float[,]
                x = new float[samples.Count, xLength],
                y = new float[samples.Count, yLength];
            for (int i = 0; i < samples.Count; i++)
            {
                Buffer.BlockCopy(samples[i].X, 0, x, sizeof(float) * xLength * i, sizeof(float) * xLength);
                Buffer.BlockCopy(samples[i].Y, 0, y, sizeof(float) * yLength * i, sizeof(float) * yLength);
            }
            return (x, y);
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
        internal static float[] CompressVertically([NotNull] this float[,] m)
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

        #endregion

        #region Argmax

        /// <summary>
        /// Returns the index of the maximum value in the input vector
        /// </summary>
        /// <param name="p">A pointer to the buffer to read</param>
        /// <param name="length">The length of the buffer to consider</param>
        [CollectionAccess(CollectionAccessType.Read)]
        internal static unsafe int Argmax(float* p, int length)
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
        public static unsafe int Argmax([NotNull] this float[] v)
        {
            fixed (float* p = v) return Argmax(p, v.Length);
        }

        /// <summary>
        /// Returns the index of the maximum value in the input matrix
        /// </summary>
        /// <param name="m">The input matrix to read from</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax([NotNull] this float[,] m)
        {
            fixed (float* p = m) return Argmax(p, m.Length);
        }

        #endregion

        #region Fill

        // Private fill method for arbitrary memory areas
        private static unsafe void Fill(float* p, int n, [NotNull] Func<float> provider)
        {
            // Fill in parallel
            int
                cores = Environment.ProcessorCount,
                batch = n / cores,
                mod = n % cores;
            Parallel.For(0, cores, i =>
            {
                int offset = i * batch;
                for (int j = 0; j < batch; j++)
                    p[offset + j] = provider();
            }).AssertCompleted();

            // Remaining values
            if (mod > 1)
                for (int i = n - mod; i < n; i++)
                    p[i] = provider();
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with the input values provider
        /// </summary>
        /// <param name="tensor">The <see cref="Tensor"/> to fill up</param>
        /// <param name="provider"></param>
        internal static unsafe void Fill(in this Tensor tensor, [NotNull] Func<float> provider) => Fill(tensor, tensor.Size, provider);

        /// <summary>
        /// Fills the target <see cref="Array"/> with the input values provider
        /// </summary>
        /// <param name="array">The <see cref="Array"/> to fill up</param>
        /// <param name="provider"></param>
        internal static unsafe void Fill([NotNull] this Array array, [NotNull] Func<float> provider)
        {
            GCHandle handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            Fill((float*)handle.AddrOfPinnedObject().ToPointer(), array.Length, provider);
            handle.Free();
        }

        #endregion

        #region Memory management

        /// <summary>
        /// Returns a deep copy of the input vector
        /// </summary>
        /// <param name="v">The vector to clone</param>
        /// <remarks>This method avoids the boxing of the <see cref="Array.Clone"/> method, and it is faster thanks to <see cref="Buffer.MemoryCopy"/></remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe float[] BlockCopy([NotNull] this float[] v)
        {
            float[] result = new float[v.Length];
            int size = sizeof(float) * v.Length;
            fixed (float* pv = v, presult = result)
                Buffer.MemoryCopy(pv, presult, size, size);
            return result;
        }

        #endregion

        #region Content check

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first <see cref="Tensor"/> to test</param>
        /// <param name="o">The second <see cref="Tensor"/> to test</param>
        /// <param name="delta">The comparison threshold</param>
        public static unsafe bool ContentEquals(in this Tensor m, in Tensor o, float delta = 1e-6f)
        {
            if (m.Ptr == IntPtr.Zero && o.Ptr == IntPtr.Zero) return true;
            if (m.Ptr == IntPtr.Zero || o.Ptr == IntPtr.Zero) return false;
            if (m.Entities != o.Entities || m.Length != o.Length) return false;
            float* pm = m, po = o;
            int items = m.Size;
            for (int i = 0; i < items; i++)
                if (!pm[i].EqualsWithDelta(po[i], delta)) return false;
            return true;
        }

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        /// <param name="delta">The comparison threshold</param>
        public static bool ContentEquals([CanBeNull] this float[,] m, [CanBeNull] float[,] o, float delta = 1e-6f)
        {
            if (m == null && o == null) return true;
            if (m == null || o == null) return false;
            if (m.GetLength(0) != o.GetLength(0) ||
                m.GetLength(1) != o.GetLength(1)) return false;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (!m[i, j].EqualsWithDelta(o[i, j], delta)) return false;
            return true;
        }

        /// <summary>
        /// Checks if two vectors have the same size and content
        /// </summary>
        /// <param name="v">The first vector to test</param>
        /// <param name="o">The second vector to test</param>
        /// <param name="delta">The comparison threshold</param>
        public static bool ContentEquals([CanBeNull] this float[] v, [CanBeNull] float[] o, float delta = 1e-6f)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (!v[i].EqualsWithDelta(o[i], delta)) return false;
            return true;
        }

        // GetUid helper method
        private static unsafe int GetUid(float* p, int n)
        {
            int hash = 17;
            unchecked
            {
                for (int i = 0; i < n; i++)
                    hash = hash * 23 + p[i].GetHashCode();
                return hash;
            }
        }

        /// <summary>
        /// Calculates a unique hash code for the target row of the input matrix
        /// </summary>
        [Pure]
        public static unsafe int GetUid([NotNull] this float[,] m, int row)
        {
            int
                w = m.GetLength(1),
                offset = row * w;
            fixed (float* pm = m)
                return GetUid(pm + offset, w);
        }

        /// <summary>
        /// Calculates a unique hash code for the input matrix
        /// </summary>
        /// <param name="m">The matrix to analyze</param>
        [Pure]
        public static unsafe int GetUid([NotNull] this float[,] m)
        {
            fixed (float* pm = m)
                return GetUid(pm, m.Length);
        }

        /// <summary>
        /// Calculates a unique hash code for the input vector
        /// </summary>
        /// <param name="v">The vector to analyze</param>
        [Pure]
        public static unsafe int GetUid([NotNull] this float[] v)
        {
            fixed (float* pv = v)
                return GetUid(pv, v.Length);
        }

        #endregion

        #region String display

        // Local helper
        [PublicAPI]
        [Pure, NotNull]
        private static unsafe String ToFormattedString(float* p, int height, int width)
        {
            if (height * width == 0) return "{ { } }";
            StringBuilder builder = new StringBuilder();
            builder.Append("{ ");
            for (int i = 0; i < height; i++)
            {
                if (width > 0)
                {
                    builder.Append("{ ");
                    for (int j = 0; j < width; j++)
                    {
                        builder.Append($"{p[i * width + j]}");
                        if (j < width - 1) builder.Append(", ");
                    }
                    builder.Append(" }");
                }
                builder.Append(i < height - 1 ? ",\n  " : " }");
            }
            return builder.ToString();
        }

        /// <summary>
        /// Returns a formatted representation of the input matrix
        /// </summary>
        /// <param name="m">The matrix to convert to <see cref="String"/></param>
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe String ToFormattedString([NotNull] this float[,] m)
        {
            fixed (float* p = m)
                return ToFormattedString(p, m.GetLength(0), m.GetLength(1));
        }

        /// <summary>
        /// Returns a formatted representation of the input matrix
        /// </summary>
        /// <param name="m">The matrix to convert to <see cref="String"/></param>
        [PublicAPI]
        [Pure, NotNull]
        public static unsafe String ToFormattedString(in this Tensor m) => ToFormattedString(m, m.Entities, m.Length);

        #endregion
    }
}
