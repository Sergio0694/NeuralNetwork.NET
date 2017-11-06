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
        /// Gets the current value for the function to optimize
        /// </summary>
        public double Cost { get; }

        /// <summary>
        /// Internal constructor for the event args base
        /// </summary>
        /// <param name="iteration">The current iteration</param>
        /// <param name="cost">The current function cost</param>
        internal BackpropagationProgressEventArgs(int iteration, double cost)
        {
            Iteration = iteration;
            Cost = cost;
        }
    }
}
