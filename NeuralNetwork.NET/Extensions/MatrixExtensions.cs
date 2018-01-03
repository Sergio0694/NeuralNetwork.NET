using System;
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
        /// Extracts a single row from a given matrix
        /// </summary>
        /// <param name="m">The source matrix</param>
        /// <param name="row">The target row to return</param>
        [PublicAPI]
        [Pure]
        public static Span<float> Slice([NotNull] this float[,] m, int row)
        {
            if (row < 0 || row > m.GetLength(0) - 1) throw new ArgumentOutOfRangeException(nameof(row), "The row index isn't valid");
            return Span<float>.DangerousCreate(m, ref m[row, 0], m.GetLength(1));
        }

        /// <summary>
        /// Finds the minimum and maximum value in the input memory area
        /// </summary>
        /// <param name="span">The memory area to scan</param>
        internal static unsafe (float Min, float Max) MinMax(in this ReadOnlySpan<float> span)
        {
            if (span.IsEmpty) return (0, 0);
            float min = float.MaxValue, max = float.MinValue;
            fixed (float* p = &span.DangerousGetPinnableReference())
            {
                for (int i = 0; i < span.Length; i++)
                {
                    float value = p[i];
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
                return (min, max);
            }
        }

        /// <summary>
        /// Returns a matrix with the reshaped content of the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The input <see cref="Span{T}"/></param>
        /// <param name="h">The number of matrix rows</param>
        /// <param name="w">The number of matrix columns</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe float[,] AsMatrix(in this Span<float> span, int h, int w)
        {
            if (h * w != span.Length) throw new ArgumentException("The input dimensions don't match the source vector size");
            float[,] m = new float[h, w];
            int size = sizeof(float) * span.Length;
            fixed (float* ps = &span.DangerousGetPinnableReference(), pm = m)
                Buffer.MemoryCopy(ps, pm, size, size);
            return m;
        }

        #endregion

        #region Argmax

        /// <summary>
        /// Returns the index of the maximum value in the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> instance</param>
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax(in this ReadOnlySpan<float> span)
        {
            if (span.Length < 2) return 0;
            int index = 0;
            float max = float.MinValue;
            fixed (float* p = &span.DangerousGetPinnableReference())
            {
                for (int j = 0; j < span.Length; j++)
                {
                    if (p[j] > max)
                    {
                        max = p[j];
                        index = j;
                    }
                }
            }
            return index;
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
