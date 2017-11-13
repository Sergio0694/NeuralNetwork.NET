using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A simple class with some extension methods
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Returns the maximum value between two numbers
        /// </summary>
        /// <param name="a">The first number</param>
        /// <param name="b">The second number</param>
        [Pure]
        public static int Max(this int a, int b) => a >= b ? a : b;

        /// <summary>
        /// Returns the minimum value between two numbers
        /// </summary>
        /// <param name="a">The first number</param>
        /// <param name="b">The second number</param>
        [Pure]
        public static int Min(this int a, int b) => a <= b ? a : b;

        /// <summary>
        /// Calculates the absolute value of the input number
        /// </summary>
        /// <param name="value">The input value</param>
        [Pure]
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
        /// Gets the maximum value among three values
        /// </summary>
        /// <param name="a">The first value <c>a</c></param>
        /// <param name="b">The second value <c>b</c></param>
        /// <param name="c">The third value <c>c</c></param>
        [Pure]
        public static float Max(float a, float b, float c)
        {
            if (a > b) return c > a ? c : a;
            return c > b ? c : b;
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
        /// Shuffles the input list using the provider <see cref="Random"/> instance
        /// </summary>
        /// <param name="list">The list to shuffle</param>
        /// <param name="random">The <see cref="Random"/> instance used to randomize the target list</param>
        public static void Shuffle<T>(this IList<T> list, [NotNull] Random random)
        {
            int n = list.Count;
            while (n > 1)
            {
                int k = random.Next(0, n) % n;
                n--;
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
