namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A structure that contains the base progress data while optimizing a network
    /// </summary>
    public sealed class BackpropagationProgressEventArgs
    {
        /// <summary>
        /// Gets the current iteration number
        /// </summary>
        public int Iteration { get; }

        /// <summary>
        /// Gets the current cost value for the network
        /// </summary>
        public float Cost { get; }

        /// <summary>
        /// Gets the current percentage of correctly classified test samples
        /// </summary>
        public float Accuracy { get; }

        /// <summary>
        /// Internal constructor for the event args base
        /// </summary>
        /// <param name="iteration">The current iteration</param>
        /// <param name="cost">The current function cost</param>
        /// <param name="accuracy">The current network accuracy</param>
        internal BackpropagationProgressEventArgs(int iteration, float cost, float accuracy)
        {
            Iteration = iteration;
            Cost = cost;
            Accuracy = accuracy;
        }
    }
}
