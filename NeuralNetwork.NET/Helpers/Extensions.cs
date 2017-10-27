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
        public static double Abs(this double value) => value >= 0 ? value : -value;

        /// <summary>
        /// Calculates if two values are within a given distance from one another
        /// </summary>
        /// <param name="value">The first value</param>
        /// <param name="other">The second value</param>
        public static bool EqualsWithDelta(this double value, double other)
        {
            if (double.IsNaN(value) ^ double.IsNaN(other)) return false;
            if (double.IsNaN(value) && double.IsNaN(other)) return true;
            if (double.IsInfinity(value) ^ double.IsInfinity(other)) return false;
            if (double.IsPositiveInfinity(value) && double.IsPositiveInfinity(other)) return true;
            if (double.IsNegativeInfinity(value) && double.IsNegativeInfinity(other)) return true;
            return (value - other).Abs() < 1e-10;
        }

        /// <summary>
        /// Gets the maximum value among three values
        /// </summary>
        /// <param name="a">The first value <c>a</c></param>
        /// <param name="b">The second value <c>b</c></param>
        /// <param name="c">The third value <c>c</c></param>
        public static double Max(double a, double b, double c)
        {
            if (a > b) return c > a ? c : a;
            return c > b ? c : b;
        }
    }
}
