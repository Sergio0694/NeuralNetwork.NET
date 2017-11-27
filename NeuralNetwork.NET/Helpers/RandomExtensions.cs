using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// Some static extension methods for the random class
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Returns the next gaussian random value (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        [PublicAPI]
        [Pure]
        public static float NextGaussian([NotNull] this Random random)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = 1.0 - random.NextDouble();
            return (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns the next boolean random value
        /// </summary>
        /// <param name="random">The random instance</param>
        [PublicAPI]
        [Pure]
        public static bool NextBool([NotNull] this Random random) => random.NextDouble() >= 0.5;

        /// <summary>
        /// Returns the next couple of indexes from within a given range (second value greater than the first one)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="n">The length of the sequence to use to generate the range</param>
        [PublicAPI]
        [Pure]
        public static (int start, int end) NextRange([NotNull] this Random random, int n)
        {
            // Checks
            if (n <= 1) throw new ArgumentOutOfRangeException("The length must be greater than 1");
            if (n == 2) return (0, 1);

            // Find the target range
            int start, end;
            do
            {
                start = random.Next(n);
                end = random.Next(n);
            } while (end <= start);
            return (start, end);
        }

        /// <summary>
        /// Returns a new vector filled with values from the Gaussian distribution (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="n">The length of the vector</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[] NextGaussianVector([NotNull] this Random random, int n)
        {
            float[] v = new float[n];
            unsafe
            {
                fixed (float* pv = v)
                    for (int i = 0; i < n; i++)
                        pv[i] = random.NextGaussian();
            }
            return v;
        }

        // Matrix initialization
        [Pure, NotNull]
        private static float[,] NextMatrix([NotNull] this Random random, int x, int y, Func<Random, float> provider)
        {
            // Checks
            if (x <= 0 || y <= 0) throw new ArgumentOutOfRangeException("The size of the matrix isn't valid");

            // Prepare the local variables
            float[,] result = new float[x, y];
            int[] seeds = new int[x];
            for (int i = 0; i < x; i++)
                seeds[i] = random.Next();

            // Populate the matrix in parallel
            Parallel.For(0, x, i =>
            {
                Random localRandom = new Random(seeds[i]);
                int offset = i * y;
                unsafe
                {
                    // Iterate over each row
                    fixed (float* r = result)
                        for (int j = 0; j < y; j++)
                            r[offset + j] = provider(localRandom);
                }
            }).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Creates a dropout mask with the given size and probability
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="dropout">The dropout probability</param>
        [Pure, NotNull]
        public static float[,] NextDropoutMask([NotNull] this Random random, int x, int y, float dropout)
        {
            float scale = 1 / dropout;
            return random.NextMatrix(x, y, r => r.NextDouble() > dropout ? scale : 0);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the Xavier initialization (with 2 / (in + out) as the variance)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <remarks>According to the implementation by Glorot & Bengio (http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)</remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextXavierMatrix([NotNull] this Random random, int x, int y)
        {
            float scale = 2f / (x + y);
            return random.NextMatrix(x, y, r => r.NextGaussian() * scale);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the sigmoid Xavier initialization (with sqrt(6 / (in + out) as the variance)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextSigmoidXavierMatrix([NotNull] this Random random, int x, int y)
        {
            float sqrt = (float)Math.Sqrt(6f / (x + y));
            return random.NextUniformMatrix(x, y, sqrt);

        }

        /// <summary>
        /// Returns a new matrix filled with random values from a random Gaussian distribution (0, 1)
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextGaussianMatrix([NotNull] this Random random, int x, int y) => random.NextMatrix(x, y, r => r.NextGaussian());

        /// <summary>
        /// Returns a matrix filled with random values in a uniform distribution of a given variance, centered in 0
        /// </summary>
        /// <param name="random">The <see cref="Random"/> instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="scale">The variance of the uniform values to generate</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextUniformMatrix([NotNull] this Random random, int x, int y, float scale)
        {
            return random.NextMatrix(x, y, r => (float)r.NextDouble() * scale * (r.NextBool() ? 1f : -1f));
        }
    }
}
