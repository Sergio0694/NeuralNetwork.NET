using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class MatrixExtensions
    {
        #region Misc

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

        #endregion

        #region Argmax

        /// <summary>
        /// Returns the index of the maximum value in the input memory area
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
        /// Returns the index of the maximum value in the input tensor
        /// </summary>
        /// <param name="tensor">The input <see cref="Tensor"/> to read from</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax(in this Tensor tensor) => Argmax(tensor, tensor.Size);

        /// <summary>
        /// Returns the index of the maximum value in the input vector
        /// </summary>
        /// <param name="v">The input vector to read from</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax([NotNull] this float[] v)
        {
            fixed (float* p = v) return Argmax(p, v.Length);
        }

        /// <summary>
        /// Returns the index of the maximum value in the input matrix
        /// </summary>
        /// <param name="m">The input matrix to read from</param>
        [PublicAPI]
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
        /// <remarks>This method avoids the boxing of the <see cref="Array.Clone"/> method, and it is faster thanks to the use of the methods in the <see cref="Buffer"/> class</remarks>
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

        #region Content equals

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first <see cref="Tensor"/> to test</param>
        /// <param name="o">The second <see cref="Tensor"/> to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static unsafe bool ContentEquals(in this Tensor m, in Tensor o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (m.Ptr == IntPtr.Zero && o.Ptr == IntPtr.Zero) return true;
            if (m.Ptr == IntPtr.Zero || o.Ptr == IntPtr.Zero) return false;
            if (m.Entities != o.Entities || m.Length != o.Length) return false;
            float* pm = m, po = o;
            int items = m.Size;
            for (int i = 0; i < items; i++)
                if (!pm[i].EqualsWithDelta(po[i], absolute, relative))
                {
                    #if DEBUG
                    if (System.Diagnostics.Debugger.IsAttached)
                        System.Diagnostics.Debug.WriteLine($"[DEBUG] {pm[i]} | {po[i]} | Threshold exceeded");
                    else Console.WriteLine($"[DEBUG] {pm[i]} | {po[i]} | Threshold exceeded");
                    #endif
                    return false;
                }
            return true;
        }

        /// <summary>
        /// Checks if two matrices have the same size and content
        /// </summary>
        /// <param name="m">The first matrix to test</param>
        /// <param name="o">The second matrix to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static bool ContentEquals([CanBeNull] this float[,] m, [CanBeNull] float[,] o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (m == null && o == null) return true;
            if (m == null || o == null) return false;
            if (m.GetLength(0) != o.GetLength(0) ||
                m.GetLength(1) != o.GetLength(1)) return false;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (!m[i, j].EqualsWithDelta(o[i, j], absolute, relative)) return false;
            return true;
        }

        /// <summary>
        /// Checks if two vectors have the same size and content
        /// </summary>
        /// <param name="v">The first vector to test</param>
        /// <param name="o">The second vector to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static bool ContentEquals([CanBeNull] this float[] v, [CanBeNull] float[] o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (v == null && o == null) return true;
            if (v == null || o == null) return false;
            if (v.Length != o.Length) return false;
            for (int i = 0; i < v.Length; i++)
                if (!v[i].EqualsWithDelta(o[i], absolute, relative)) return false;
            return true;
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
