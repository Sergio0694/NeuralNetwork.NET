using System;
using System.Threading.Tasks;
using JetBrains.Annotations;

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
        /// <param name="random">The random instance</param>
        [PublicAPI]
        public static float NextGaussian([NotNull] this Random random)
        {
            double u1 = random.NextDouble(), u2 = random.NextDouble();
            return (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns the next boolean random value
        /// </summary>
        /// <param name="random">The random instance</param>
        [PublicAPI]
        public static bool NextBool([NotNull] this Random random) => random.NextDouble() >= 0.5;

        /// <summary>
        /// Returns the next couple of indexes from within a given range (second value greater than the first one)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="n">The length of the sequence to use to generate the range</param>
        [PublicAPI]
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
        /// <param name="random">The random instance</param>
        /// <param name="n">The length of the vector</param>
        [PublicAPI]
        [NotNull]
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
        public static float[,] NextMatrix([NotNull] this Random random, int x, int y, Func<Random, float> provider)
        {
            // Checks
            if (x <= 0 || y <= 0) throw new ArgumentOutOfRangeException("The size of the matrix isn't valid");

            // Prepare the local variables
            float[,] result = new float[x, y];
            int[] seeds = new int[x];
            for (int i = 0; i < x; i++)
                seeds[i] = random.Next();

            // Populate the matrix in parallel
            bool loopResult = Parallel.For(0, x, i =>
            {
                Random localRandom = new Random(seeds[i]);
                unsafe
                {
                    // Iterate over each row
                    fixed (float* r = result)
                        for (int j = 0; j < y; j++)
                            r[i * y + j] = provider(localRandom);
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while running the parallel loop");
            return result;
        }

        /// <summary>
        /// Returns a new matrix filled with values from the Xavier initialization (random~N(0,1) over the square of the number of input neurons)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        [PublicAPI]
        [NotNull]
        public static float[,] NextXavierMatrix([NotNull] this Random random, int x, int y)
        {
            float sqrt = (float)Math.Sqrt(x);
            return random.NextMatrix(x, y, r => r.NextGaussian() / sqrt);
        }

        /// <summary>
        /// Returns a new matrix filled with random values from a random Gaussian distribution (0, 1)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        [PublicAPI]
        [NotNull]
        public static float[,] NextGaussianMatrix([NotNull] this Random random, int x, int y) => random.NextMatrix(x, y, r => r.NextGaussian());
    }
}
