namespace NeuralNetworkNET.UnsupervisedLearning.Misc
{
    /// <summary>
    /// Contains the info on an evaluated generation
    /// </summary>
    public struct GeneticAlgorithmProgress
    {
        /// <summary>
        /// The generation number
        /// </summary>
        public int Generation { get; }

        /// <summary>
        /// The best score for the generation
        /// </summary>
        public double Best { get; }

        /// <summary>
        /// The average score for the generation
        /// </summary>
        public double Average { get; }

        /// <summary>
        /// The best score since the start of the genetic algorithm
        /// </summary>
        public double AllTimeBest { get; }

        // Internal constructor
        internal GeneticAlgorithmProgress(int generation, double best, double average, double allTime)
        {
            Generation = generation;
            Best = best;
            Average = average;
            AllTimeBest = allTime;
        }
    }
}
