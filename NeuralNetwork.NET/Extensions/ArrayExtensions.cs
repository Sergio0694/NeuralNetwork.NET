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
        /// Merges the line pairs in the input collection into two 2D arrays
        /// </summary>
        /// <typeparam name="T">The type of each element in the input lines</typeparam>
        /// <param name="lines">The lines to merge</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static (T[,], T[,]) MergeLines<T>(this IEnumerable<(T[], T[])> lines) where T : struct
        {
            (T[] X, T[] Y)[] set = lines.ToArray();
            T[,] 
                x = new T[set.Length, set[0].X.Length],
                y = new T[set.Length, set[0].Y.Length];
            Parallel.For(0, set.Length, i =>
            {
                set[i].X.AsSpan().CopyTo(x.Slice(i));
                set[i].Y.AsSpan().CopyTo(y.Slice(i));
            }).AssertCompleted();
            return (x, y);
        }
    }
}
