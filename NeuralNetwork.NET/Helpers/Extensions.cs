using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkNET.Helpers
{
    public static class Extensions
    {
        public static double Abs(this double value) => value >= 0 ? value : -value;

        private const double DefaultThreshold = 0.00001;

        public static bool EqualsWithDelta(this double value, double other, double threshold = DefaultThreshold)
        {
            if (double.IsNaN(value) ^ double.IsNaN(other)) return false;
            if (double.IsNaN(value) && double.IsNaN(other)) return true;
            if (double.IsInfinity(value) ^ double.IsInfinity(other)) return false;
            if (double.IsPositiveInfinity(value) && double.IsPositiveInfinity(other)) return true;
            if (double.IsNegativeInfinity(value) && double.IsNegativeInfinity(other)) return true;
            return (value - other).Abs() < threshold;
        }
    }
}
