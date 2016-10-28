namespace System
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
        public static double NextGaussian(this Random random)
        {
            double u1 = random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Returns the next boolean random value
        /// </summary>
        /// <param name="random">The random instance</param>
        public static bool NextBool(this Random random) => random.Next() % 2 == 0;

        /// <summary>
        /// Returns the next couple of indexes from within a given range (second value greater than the first one)
        /// </summary>
        /// <param name="random">The random instance</param>
        /// <param name="n">The length of the sequence to use to generate the range</param>
        public static Range NextRange(this Random random, int n)
        {
            int start, end;
            do
            {
                start = random.Next(n);
                end = random.Next(n);
            } while (end <= start);
            return new Range(start, end);
        }
    }

    /// <summary>
    /// A struct that represents a range from two integer values
    /// </summary>
    public struct Range
    {
        /// <summary>
        /// Gets the start of the range
        /// </summary>
        public int Start { get; }

        /// <summary>
        /// Gets the end of the range
        /// </summary>
        public int End { get; }

        // Internal constructor
        internal Range(int start, int end)
        {
            if (end <= start) throw new ArgumentOutOfRangeException();
            Start = start;
            End = end;
        }
    }
}
