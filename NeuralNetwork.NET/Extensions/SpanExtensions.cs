using System;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// An helper class with methods to process fixed-size matrices
    /// </summary>
    public static class SpanExtensions
    {
        #region Generics

        /// <summary>
        /// Fills the target <see cref="Span{T}"/> with the input values provider
        /// </summary>
        /// <param name="span">The <see cref="Span{T}"/> to fill up</param>
        /// <param name="provider">The values provider to use</param>
        public static unsafe void Fill<T>(this Span<T> span, [NotNull] Func<T> provider) where T : unmanaged
        {
            // Fill in parallel
            int
                cores = Environment.ProcessorCount,
                batch = span.Length / cores,
                mod = span.Length % cores;
            fixed (T* p0 = span)
            {
                T* pc = p0;
                Parallel.For(0, cores, i =>
                {
                    T* p = pc + i * batch;
                    for (int j = 0; j < batch; j++)
                        p[j] = provider();
                }).AssertCompleted();

                // Remaining values
                if (mod < 1) return;
                for (int i = span.Length - mod; i < span.Length; i++)
                    pc[i] = provider();
            }
        }

        /// <summary>
        /// Returns a matrix with the reshaped content of the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The input <see cref="Span{T}"/></param>
        /// <param name="h">The number of matrix rows</param>
        /// <param name="w">The number of matrix columns</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe T[,] AsMatrix<T>(this Span<T> span, int h, int w) where T : unmanaged
        {
            if (h * w != span.Length) throw new ArgumentException("The input dimensions don't match the source vector size");
            T[,] m = new T[h, w];
            fixed (void* p = m)
                span.CopyTo(new Span<T>(p, m.Length));
            return m;
        }

        /// <summary>
        /// Returns an hash code for the contents of the input <see cref="Span{T}"/>
        /// </summary>
        /// <typeparam name="T">The type of each value in the input <see cref="Span{T}"/></typeparam>
        /// <param name="span">The input <see cref="Span{T}"/> to read</param>
        [Pure]
        public static unsafe int GetContentHashCode<T>(this Span<T> span) where T : unmanaged
        {
            fixed (T* p0 = span)
            {
                int hash = 17;
                unchecked
                {
                    for (int i = 0; i < span.Length; i++)
                        hash = hash * 23 + p0[i].GetHashCode();
                    return hash;
                }
            }
        }

        /// <summary>
        /// Extracts a single row from a given matrix
        /// </summary>
        /// <param name="m">The source matrix</param>
        /// <param name="row">The target row to return</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe T[] Slice<T>([NotNull] this T[,] m, int row) where T : unmanaged
        {
            if (row < 0 || row > m.GetLength(0) - 1) throw new ArgumentOutOfRangeException(nameof(row), "The row index isn't valid");
            int wm = m.GetLength(1);
            fixed (T* p = m) return new Span<T>(p + row * wm, wm).ToArray();
        }

        #endregion

        #region Float

        /// <summary>
        /// Returns the index of the maximum value in the input <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The source <see cref="Span{T}"/> instance</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe int Argmax(this Span<float> span)
        {
            if (span.Length < 2) return default;
            int index = 0;
            float max = float.MinValue;
            fixed (float* p = span)
            {
                for (int j = 0; j < span.Length; j++)
                {
                    float value = p[j];
                    if (value > max)
                    {
                        max = value;
                        index = j;
                    }
                }
            }
            return index;
        }

        /// <summary>
        /// Returns whether or not all the elements in the two input <see cref="Span{T}"/> instances respect the input threshold
        /// </summary>
        /// <param name="x1">The first <see cref="Span{T}"/> instance to check</param>
        /// <param name="x2">The second <see cref="Span{T}"/> instance to check</param>
        /// <param name="threshold">The target threshold</param>
        /// <remarks>This method is <see langword="internal"/> as it's meant to be exposed through the <see cref="APIs.Settings.AccuracyTesters"/> class only</remarks>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal static unsafe bool MatchElementwiseThreshold(this Span<float> x1, Span<float> x2, float threshold)
        {
            if (x1.Length != x2.Length) throw new ArgumentException("The two input spans must have the same length");
            fixed (float* px1 = x1, px2 = x2)
                for (int i = 0; i < x1.Length; i++)
                    if (px1[i] > threshold != px2[i] > threshold)
                        return false;
            return true;
        }

        /// <summary>
        /// Returns whether or not all the elements in the two input <see cref="Span{T}"/> respect the maximum distance between each other
        /// </summary>
        /// <param name="x1">The first <see cref="Span{T}"/> instance to check</param>
        /// <param name="x2">The second <see cref="Span{T}"/> instance to check</param>
        /// <param name="threshold">The target maximum distance</param>
        /// <remarks>This method is <see langword="internal"/> as it's meant to be exposed through the <see cref="APIs.Settings.AccuracyTesters"/> class only</remarks>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal static unsafe bool IsCloseTo(this Span<float> x1, Span<float> x2, float threshold)
        {
            if (x1.Length != x2.Length) throw new ArgumentException("The two input spans must have the same length");
            fixed (float* px1 = x1, px2 = x2)
                for (int i = 0; i < x1.Length; i++)
                    if ((px1[i] - px2[i]).Abs() > threshold)
                        return false;
            return true;
        }

        #endregion
    }
}
