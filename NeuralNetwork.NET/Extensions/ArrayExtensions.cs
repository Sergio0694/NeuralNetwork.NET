using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A static class with some general methods for linear and 2D arrays
    /// </summary>
    internal static class ArrayExtensions
    {
        /// <summary>
        /// Flattens a 2D array to a 1D array
        /// </summary>
        /// <typeparam name="T">The type of each element in the input matrix</typeparam>
        /// <param name="m">The input 2D array to flatten</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe T[] Flatten<T>([NotNull] this T[,] m) where T : unmanaged
        {
            fixed (T* p = m) return new Span<T>(p, m.Length).ToArray();
        }

        /// <summary>
        /// Merges the line pairs in the input collection into two 2D arrays
        /// </summary>
        /// <typeparam name="T">The type of each element in the input lines</typeparam>
        /// <param name="lines">The lines to merge</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe (T[,], T[,]) MergeLines<T>(this IEnumerable<(T[], T[])> lines) where T : unmanaged
        {
            (T[] X, T[] Y)[] set = lines.ToArray();
            T[,] 
                x = new T[set.Length, set[0].X.Length],
                y = new T[set.Length, set[0].Y.Length];
            int
                wx = x.GetLength(1),
                wy = y.GetLength(1);
            fixed (T* px0 = x, py0 = y)
            {
                T* px = px0, py = py0;
                Parallel.For(0, set.Length, i =>
                {
                    set[i].X.AsSpan().CopyTo(new Span<T>(px + i * wx, wx));
                    set[i].Y.AsSpan().CopyTo(new Span<T>(py + i * wy, wy));
                }).AssertCompleted();
            }
            return (x, y);
        }
    }
}
