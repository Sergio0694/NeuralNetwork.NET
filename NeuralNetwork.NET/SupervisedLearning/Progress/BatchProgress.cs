namespace NeuralNetworkNET.SupervisedLearning.Progress
{
    /// <summary>
    /// A simple <see langword="struct"/> containing info on the current batch progress for the training dataset
    /// </summary>
    public readonly struct BatchProgress
    {
        /// <summary>
        /// Gets the current total number of processed samples
        /// </summary>
        public readonly int ProcessedItems;

        /// <summary>
        /// Gets the current training dataset progress percentage
        /// </summary>
        public readonly float Percentage;

        // Internal constructor for the network trainer
        internal BatchProgress(int processed, float percentage)
        {
            ProcessedItems = processed;
            Percentage = percentage;
        }
    }
}
