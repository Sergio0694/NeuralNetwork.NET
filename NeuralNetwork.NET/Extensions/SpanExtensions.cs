using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class SpanExtensions
    {
        #region Generic extensions

        /// <summary>
        /// Extracts a single row from a given matrix
        /// </summary>
        /// <param name="m">The source matrix</param>
        /// <param name="row">The target row to return</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Span<T> Slice<T>([NotNull] this T[,] m, int row) where T : struct
        {
            if (row < 0 || row > m.GetLength(0) - 1) throw new ArgumentOutOfRangeException(nameof(row), "The row index isn't valid");
            return Span<T>.DangerousCreate(m, ref m[row, 0], m.GetLength(1));
        }

        /// <summary>
        /// Finds the minimum and maximum value in the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> to scan</param>
        /// <param name="min">The minimum possible value for a <see cref="T"/> value</param>
        /// <param name="max">The maaximum possible value for a <see cref="T"/> value</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static (T Min, T Max) MinMax<T>(this Span<T> span, T min, T max) where T : struct, IComparable<T>
        {
            if (span.IsEmpty) return (default, default);
            T low = max, high = min;
            for (int i = 0; i < span.Length; i++)
            {
                T value = span[i];
                if (value.CompareTo(low) < 0) low = value;
                if (value.CompareTo(high) > 0) high = value;
            }
            return (low, high);
        }

        /// <summary>
        /// Returns a matrix with the reshaped content of the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The input <see cref="Span{T}"/></param>
        /// <param name="h">The number of matrix rows</param>
        /// <param name="w">The number of matrix columns</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static T[,] AsMatrix<T>(this Span<T> span, int h, int w) where T : struct
        {
            if (h * w != span.Length) throw new ArgumentException("The input dimensions don't match the source vector size");
            T[,] m = new T[h, w];
            span.CopyTo(Span<T>.DangerousCreate(m, ref m[0, 0], m.Length));
            return m;
        }

        /// <summary>
        /// Returns the index of the maximum value in the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> instance</param>
        /// <param name="min">The minimum possible value for a <see cref="T"/> value</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static int Argmax<T>(this Span<T> span, T min) where T : struct, IComparable<T>
        {
            if (span.Length < 2) return default;
            int index = 0;
            T max = min;
            for (int j = 0; j < span.Length; j++)
            {
                T value = span[j];
                if (value.CompareTo(max) > 0)
                {
                    max = value;
                    index = j;
                }
            }
            return index;
        }

        /// <summary>
        /// Returns a deep copy of the input vector
        /// </summary>
        /// <param name="v">The vector to clone</param>
        /// <remarks>This method avoids the boxing of the <see cref="Array.Clone"/> method, and it is faster thanks to the use of the methods in the <see cref="Buffer"/> class</remarks>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static T[] BlockCopy<T>([NotNull] this T[] v) where T : struct
        {
            T[] result = new T[v.Length];
            Buffer.BlockCopy(v, 0, result, 0, Unsafe.SizeOf<T>() * result.Length);
            return result;
        }

        /// <summary>
        /// Returns a new <see cref="Span{T}"/> instance from the input matrix
        /// </summary>
        /// <typeparam name="T">The type of the input matrix</typeparam>
        /// <param name="m">The input matrix</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<T> AsSpan<T>([NotNull] this T[,] m) where T : struct => Span<T>.DangerousCreate(m, ref m[0, 0], m.Length);

        #endregion

        #region Float

        /// <summary>
        /// Fills the target <see cref="Span{T}"/> with the input values provider
        /// </summary>
        /// <param name="span">The <see cref="Span{T}"/> to fill up</param>
        /// <param name="provider">The values provider to use</param>
        public static unsafe void Fill(this Span<float> span, [NotNull] Func<float> provider)
        {
            // Fill in parallel
            int
                cores = Environment.ProcessorCount,
                batch = span.Length / cores,
                mod = span.Length % cores;
            fixed (float* p = &span.DangerousGetPinnableReference())
            {
                float* p0 = p;
                Parallel.For(0, cores, i =>
                {
                    int offset = i * batch;
                    for (int j = 0; j < batch; j++)
                        p0[offset + j] = provider();
                }).AssertCompleted();

                // Remaining values
                if (mod > 1)
                    for (int i = span.Length - mod; i < span.Length; i++)
                        p[i] = provider();
            }
        }

        /// <summary>
        /// Checks if two <see cref="Span{T}"/> instances have the same size and content
        /// </summary>
        /// <param name="x1">The first <see cref="Span{T}"/> to test</param>
        /// <param name="x2">The second <see cref="Span{T}"/> to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static unsafe bool ContentEquals(this Span<float> x1, Span<float> x2, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (x1.Length != x2.Length) return false;
            fixed (float* p1 = &x1.DangerousGetPinnableReference(), p2 = &x2.DangerousGetPinnableReference())
            {
                for (int i = 0; i < x1.Length; i++)
                    if (!p1[i].EqualsWithDelta(p2[i], absolute, relative)) return false;
            }
            return true;
        }

        /// <summary>
        /// Checks if two <see cref="Tensor"/> instances have the same size and content
        /// </summary>
        /// <param name="m">The first <see cref="Tensor"/> to test</param>
        /// <param name="o">The second <see cref="Tensor"/> to test</param>
        /// <param name="absolute">The relative comparison threshold</param>
        /// <param name="relative">The relative comparison threshold</param>
        public static bool ContentEquals(in this Tensor m, in Tensor o, float absolute = 1e-6f, float relative = 1e-6f)
        {
            if (m.Ptr == IntPtr.Zero && o.Ptr == IntPtr.Zero) return true;
            if (m.Ptr == IntPtr.Zero || o.Ptr == IntPtr.Zero) return false;
            if (m.Entities != o.Entities || m.Length != o.Length) return false;
            return m.AsSpan().ContentEquals(o.AsSpan(), absolute, relative);
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
            return m.AsSpan().ContentEquals(o.AsSpan(), absolute, relative);
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
            return v.AsSpan().ContentEquals(o, absolute, relative);
        }

        #endregion
    }
}
