using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A simple class with some extension methods
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Casts the input item to a class or interface that inherits from the initial type
        /// </summary>
        /// <typeparam name="TIn">The input type</typeparam>
        /// <typeparam name="TOut">The output type</typeparam>
        /// <param name="item">The item to cast</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TOut To<TIn, TOut>([NotNull] this TIn item) where TOut : class, TIn => item as TOut 
            ?? throw new InvalidOperationException($"The item of type {typeof(TIn)} is a {item.GetType()} instance and can't be cast to {typeof(TOut)}");

        /// <summary>
        /// Returns the maximum value between two numbers
        /// </summary>
        /// <param name="a">The first number</param>
        /// <param name="b">The second number</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Max(this int a, int b) => a >= b ? a : b;

        /// <summary>
        /// Returns the minimum value between two numbers
        /// </summary>
        /// <param name="a">The first number</param>
        /// <param name="b">The second number</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Min(this int a, int b) => a <= b ? a : b;

        /// <summary>
        /// Calculates the absolute value of the input number
        /// </summary>
        /// <param name="value">The input value</param>
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Abs(this float value) => value >= 0 ? value : -value;

        /// <summary>
        /// Calculates if two values are within a given distance from one another
        /// </summary>
        /// <param name="value">The first value</param>
        /// <param name="other">The second value</param>
        /// <param name="delta">The comparison threshold</param>
        [Pure]
        public static bool EqualsWithDelta(this float value, float other, float delta = 1e-6f)
        {
            if (float.IsNaN(value) ^ float.IsNaN(other)) return false;
            if (float.IsNaN(value) && float.IsNaN(other)) return true;
            if (float.IsInfinity(value) ^ float.IsInfinity(other)) return false;
            if (float.IsPositiveInfinity(value) && float.IsPositiveInfinity(other)) return true;
            if (float.IsNegativeInfinity(value) && float.IsNegativeInfinity(other)) return true;
            return (value - other).Abs() < delta;
        }

        /// <summary>
        /// Calculates the integer square of the input value
        /// </summary>
        /// <param name="x">The value to use to calculate the square root</param>
        [Pure]
        public static int IntegerSquare(this int x)
        {
            if (x < 0) throw new ArgumentOutOfRangeException(nameof(x), "The input value can't be negative");
            if (0 == x) return 0;   // Avoid division by zero 
            int n = x / 2 + 1;      // Initial estimate, this is never too low 
            int n1 = (n + x / n) / 2;
            while (n1 < n)
            {
                n = n1;
                n1 = (n + x / n) / 2;
            }
            return n;
        }

        /// <summary>
        /// Rounds the given <see cref="TimeSpan"/> to an interval with an integer number of total seconds
        /// </summary>
        /// <param name="timeSpan">The instance to round</param>
        [Pure]
        public static TimeSpan RoundToSeconds(this TimeSpan timeSpan) => TimeSpan.FromSeconds((int)Math.Floor(timeSpan.TotalSeconds));

        /// <summary>
        /// Partitions the input sequence into a series of batches of the given size
        /// </summary>
        /// <typeparam name="T">The type of the sequence items</typeparam>
        /// <param name="values">The sequence of items to batch</param>
        /// <param name="size">The desired batch size</param>
        [PublicAPI]
        [Pure, NotNull, ItemNotNull]
        public static IEnumerable<IReadOnlyList<T>> Partition<T>([NotNull] this IEnumerable<T> values, int size)
        {
            // Private batch enumerator
            IEnumerable<T> GetChunk(IEnumerator<T> enumerator)
            {
                int n = size;
                do yield return enumerator.Current;
                while (--n > 0 && enumerator.MoveNext());
            }

            // Enumerate the sequence and partition the batches
            using (IEnumerator<T> enumerator = values.GetEnumerator())
                while (enumerator.MoveNext())
                    yield return GetChunk(enumerator).ToArray();
        }
    }
}
