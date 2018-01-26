using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.APIs.Settings
{
    /// <summary>
    /// A static class exposing different <see cref="AccuracyTester"/> options to monitor the accuracy of a neural network
    /// </summary>
    public static class AccuracyTesters
    {
        /// <summary>
        /// Gets an <see cref="AccuracyTester"/> <see langword="delegate"/> that can be used classification problems with mutually-exclusive classes
        /// </summary>
        [PublicAPI]
        [Pure, NotNull]
        public static AccuracyTester Argmax() => (yHat, y) => yHat.Argmax() == y.Argmax();

        /// <summary>
        /// Gets an <see cref="AccuracyTester"/> <see langword="delegate"/> that checks if all the output values match the expected threshold
        /// </summary>
        [PublicAPI]
        [Pure, NotNull]
        public static AccuracyTester Threshold(float threshold = 0.5f)
        {
            if (threshold <= 0 || threshold >= 1) throw new ArgumentOutOfRangeException(nameof(threshold), "The threshold must be in the (0,1) range");
            return (yHat, y) => yHat.MatchElementwiseThreshold(y, threshold);
        }

        /// <summary>
        /// Gets an <see cref="AccuracyTester"/> <see langword="delegate"/> that checks if all the output values are within a specific distance from the expected values
        /// </summary>
        [PublicAPI]
        [Pure, NotNull]
        public static AccuracyTester Distance(float distance)
        {
            if (distance <= 0) throw new ArgumentOutOfRangeException(nameof(distance), "The distance parameter must be a positive value");
            return (yHat, y) => yHat.IsCloseTo(y, distance);
        }
    }
}
