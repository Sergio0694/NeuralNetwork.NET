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
        public static double NextGaussian([NotNull] this Random random)
        {
            double u1 = random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns the next boolean random value
        /// </summary>
        /// <param name="random">The random instance</param>
        [PublicAPI]
        public static bool NextBool([NotNull] this Random random) => random.Next() % 2 == 0;

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
        /// Returns a new matrix filled with random values from a random Gaussian distribution (0, 1)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        [PublicAPI]
        [NotNull]
        public static double[,] NextMatrix([NotNull] this Random random, int x, int y)
        {
            // Checks
            if (x <= 0 || y <= 0) throw new ArgumentOutOfRangeException("The size of the matrix isn't valid");

            // Prepare the local variables
            double[,] result = new double[x, y];
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
                    fixed (double* r = result)
                        for (int j = 0; j < y; j++)
                            r[i * y + j] = localRandom.NextGaussian();
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while running the parallel loop");
            return result;
        }

        /// <summary>
        /// Randomizes part of the content of a matrix
        /// </summary>
        /// <param name="m">The matrix to randomize</param>
        /// <param name="probability">The probabiity of each matrix element to be randomized</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Randomize([NotNull] this double[,] m, double probability)
        {
            if (probability < 0 || probability > 1) throw new ArgumentOutOfRangeException("The probability must be in the [0, 1] range");
            double inverse = 1.0 - probability;
            int h = m.GetLength(0), w = m.GetLength(1);
            double[,] randomized = new double[h, w];
            bool loopResult = Parallel.For(0, m.GetLength(0), i =>
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
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while runnig the parallel loop");
            return randomized;
        }

        /// <summary>
        /// Randomizes part of the content of a vector
        /// </summary>
        /// <param name="v">The vector to randomize</param>
        /// <param name="probability">The probabiity of each vector element to be randomized</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] Randomize([NotNull] this double[] v, double probability)
        {
            // Checks and local parameters
            if (probability < 0 || probability > 1) throw new ArgumentOutOfRangeException("The probability must be in the [0, 1] range");
            double inverse = 1.0 - probability;
            double[] randomized = new double[v.Length];
            Random random = new Random();

            // Populate the resulting vector
            unsafe
            {
                fixed (double* r = randomized, pv = v)
                    for (int i = 0; i < v.Length; i++)
                    {
                        if (random.NextDouble() >= inverse)
                            r[i] = random.NextDouble();
                        else r[i] = pv[i];
                    }
            }
            return randomized;
        }

        /// <summary>
        /// Performs a random two points crossover between two matrices
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal static double[,] TwoPointsCrossover([NotNull] this Random random, [NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            // Get the size of the matrix and check the input
            int h = m1.GetLength(0), w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentOutOfRangeException();

            // Prepare the local variables
            (int Start, int End) xr = random.NextRange(h), yr = random.NextRange(w);
            double[,] result = new double[h, w];

            // Populate the matrix in parallel
            bool loopResult = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    // Iterate over each row
                    fixed (double* r = result, p1 = m1, p2 = m2)
                        for (int j = 0; j < w; j++)
                        {
                            // Perform the crossover when needed
                            if (i >= xr.Start && i <= xr.End && j >= yr.Start && j <= yr.End)
                            {
                                r[i * w + j] = p2[i * w + j];
                            }
                            else r[i * w + j] = p1[i * w + j];
                        }
                }
            }).IsCompleted;
            if (!loopResult) throw new Exception("Error while running the parallel loop");
            return result;
        }
    }
}
