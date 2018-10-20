using System;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A thread-safe <see cref="Random"/> wrapper class with some extension methods
    /// </summary>
    public static class ThreadSafeRandom
    {
        // Shared seed
        private static int _Seed = Environment.TickCount;

        // Random instance that provides the pseudo-random numbers
        private static readonly ThreadLocal<Random> RandomInstance = new ThreadLocal<Random>(() => new Random(Interlocked.Increment(ref _Seed)));

        #region Base extensions

        /// <summary>
        /// Returns the next uniform value with the desired range
        /// </summary>
        /// <param name="scale">The desired scale for the uniform distribution</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextUniform(float scale = 1)
        {
            return (float)RandomInstance.Value.NextDouble() * scale * (NextBool() ? 1f : -1f);
        }

        /// <summary>
        /// Returns the next gaussian random value (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="deviation">The standard deviation for the distribution</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextGaussian(float deviation = 1)
        {
            double u1 = 1.0 - RandomInstance.Value.NextDouble(), u2 = 1.0 - RandomInstance.Value.NextDouble();
            return (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2) * deviation;
        }

        /// <summary>
        /// Returns the next <see cref="int"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int NextInt(int min = 0, int max = int.MaxValue) => RandomInstance.Value.Next(min, max);

        /// <summary>
        /// Returns the next <see cref="float"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextFloat() => (float)RandomInstance.Value.NextDouble();

        /// <summary>
        /// Returns the next <see cref="bool"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool NextBool() => RandomInstance.Value.NextDouble() >= 0.5;

        #endregion

        #region Misc

        /// <summary>
        /// Creates a dropout mask with the given size and probability
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="dropout">The dropout probability (indicates the probability of keeping a neuron active)</param>
        /// <param name="mask">The resulting mask</param>
        public static unsafe void NextDropoutMask(int x, int y, float dropout, out Tensor mask)
        {
            if (x <= 0 || y <= 0) throw new ArgumentOutOfRangeException(nameof(x), "The size of the matrix isn't valid");
            float scale = 1 / dropout;
            Tensor.New(x, y, out mask);
            float* r = mask;
            Parallel.For(0, x, i =>
            {
                int offset = i * y;
                for (int j = 0; j < y; j++)
                    r[offset + j] = NextFloat() > dropout ? 0 : scale;
            }).AssertCompleted();
        }

        #endregion
    }
}
