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
        public float Best { get; }

        /// <summary>
        /// The average score for the generation
        /// </summary>
        public float Average { get; }

        /// <summary>
        /// The best score since the start of the genetic algorithm
        /// </summary>
        public float AllTimeBest { get; }

        // Internal constructor
        internal GeneticAlgorithmProgress(int generation, float best, float average, float allTime)
        {
            Generation = generation;
            Best = best;
            Average = average;
            AllTimeBest = allTime;
        }
    }
}
