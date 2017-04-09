using System;
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
        public static double NextGaussian([NotNull] this Random random)
        {
            double u1 = random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns the next boolean random value
        /// </summary>
        /// <param name="random">The random instance</param>
        public static bool NextBool([NotNull] this Random random) => random.Next() % 2 == 0;

        /// <summary>
        /// Returns the next couple of indexes from within a given range (second value greater than the first one)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="n">The length of the sequence to use to generate the range</param>
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
        /// Randomizes part of the content of a matrix
        /// </summary>
        /// <param name="m">The matrix to randomize</param>
        /// <param name="probability">The probabiity of each matrix element to be randomized</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Randomize([NotNull] this double[,] m, double probability)
        {
            if (probability < 0 || probability > 1) throw new ArgumentOutOfRangeException("The probability must be in the [0, 1] range");
            double inverse = 1.0 - probability;
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] randomized = new double[h, w];
            bool loopResult = ParallelCompatibilityWrapper.Instance.Invoke(0, m.GetLength(0), i =>
            {
                // Get the random instance and fix the pointers
                Random random = new Random();
                unsafe
                {
                    fixed (double* r = randomized, pm = m)
                    {
                        // Populate the resulting matrix
                        for (int j = 0; j < w; j++)
                        {
                            if (random.NextDouble() >= inverse)
                                r[i * w + j] = random.NextDouble();
                            else r[i * w + j] = pm[i * w + j];
                        }
                    }
                }
            });
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return randomized;
        }

        /// <summary>
        /// Randomizes part of the content of a vector
        /// </summary>
        /// <param name="v">The vector to randomize</param>
        /// <param name="probability">The probabiity of each vector element to be randomized</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Randomize([NotNull] this double[] v, double probability)
        {
            // Checks
            if (probability < 0 || probability > 1) throw new ArgumentOutOfRangeException("The probability must be in the [0, 1] range");
            double inverse = 1.0 - probability;
            double[] randomized = new double[v.Length];

            // Populate the resulting vector
            unsafe
            {
                fixed (double* r = randomized, pv = v)
                {
                    for (int i = 0; i < v.Length; i++)
                    {
                        Random random = new Random();
                        if (random.NextDouble() >= inverse)
                            r[i] = random.NextDouble();
                        else r[i] = pv[i];
                    }
                }
            }
            return randomized;
        }
    }
}
