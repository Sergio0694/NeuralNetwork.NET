using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A simple class with some extension methods
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Calculates the absolute value of the input number
        /// </summary>
        /// <param name="value">The input value</param>
        [Pure]
        public static double Abs(this double value) => value >= 0 ? value : -value;

        /// <summary>
        /// Calculates if two values are within a given distance from one another
        /// </summary>
        /// <param name="value">The first value</param>
        /// <param name="other">The second value</param>
        /// <param name="delta">The comparison threshold</param>
        [Pure]
        public static bool EqualsWithDelta(this double value, double other, double delta = 1e-10)
        {
            if (double.IsNaN(value) ^ double.IsNaN(other)) return false;
            if (double.IsNaN(value) && double.IsNaN(other)) return true;
            if (double.IsInfinity(value) ^ double.IsInfinity(other)) return false;
            if (double.IsPositiveInfinity(value) && double.IsPositiveInfinity(other)) return true;
            if (double.IsNegativeInfinity(value) && double.IsNegativeInfinity(other)) return true;
            return (value - other).Abs() < delta;
        }

        /// <summary>
        /// Gets the maximum value among three values
        /// </summary>
        /// <param name="a">The first value <c>a</c></param>
        /// <param name="b">The second value <c>b</c></param>
        /// <param name="c">The third value <c>c</c></param>
        [Pure]
        public static double Max(double a, double b, double c)
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
    }
}
