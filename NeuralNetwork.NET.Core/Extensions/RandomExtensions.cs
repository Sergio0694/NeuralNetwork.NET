using System;
using JetBrains.Annotations;

namespace NeuralNetworkDotNet.Core.Extensions
{
    /// <summary>
    /// An helper <see langword="class"/> with extension methods for the <see cref="Random"/> type
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Returns the next uniform value with the desired range
        /// </summary>
        /// <param name="random">The input <see cref="Random"/> instance to use</param>
        /// <param name="scale">The desired scale for the uniform distribution</param>
        [Pure]
        public static float NextUniform([NotNull] this Random random, float scale = 1)
        {
            return (float)random.NextDouble() * scale * (random.NextBool() ? 1f : -1f);
        }

        /// <summary>
        /// Returns the next gaussian random value (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="random">The input <see cref="Random"/> instance to use</param>
        /// <param name="deviation">The standard deviation for the distribution</param>
        [Pure]
        public static float NextGaussian([NotNull] this Random random, float deviation = 1)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = 1.0 - random.NextDouble();
            return (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2) * deviation;
        }

        /// <summary>
        /// Returns the next <see cref="int"/> random value
        /// </summary>
        /// <param name="random">The input <see cref="Random"/> instance to use</param>
        /// <param name="min">The minimum value</param>
        /// <param name="max">The maximum value (excluded)</param>
        [Pure]
        public static int NextInt([NotNull] this Random random, int min = 0, int max = int.MaxValue) => random.Next(min, max);

        /// <summary>
        /// Returns the next <see cref="float"/> random value
        /// </summary>
        /// <param name="random">The input <see cref="Random"/> instance to use</param>
        [Pure]
        public static float NextFloat([NotNull] this Random random) => (float)random.NextDouble();

        /// <summary>
        /// Returns the next <see cref="bool"/> random value
        /// </summary>
        /// <param name="random">The input <see cref="Random"/> instance to use</param>
        [Pure]
        public static bool NextBool([NotNull] this Random random) => random.NextDouble() >= 0.5;
    }
}
