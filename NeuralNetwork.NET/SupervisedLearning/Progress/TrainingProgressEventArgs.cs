using System;
using NeuralNetworkNET.APIs.Results;

namespace NeuralNetworkNET.SupervisedLearning.Progress
{
    /// <summary>
    /// A structure that contains the base progress data while optimizing a network
    /// </summary>
    public sealed class TrainingProgressEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the current iteration number
        /// </summary>
        public int Iteration { get; }

        /// <summary>
        /// Gets the current cost value for the network
        /// </summary>
        public DatasetEvaluationResult Result { get; }

        /// <summary>
        /// Internal constructor for the event args base
        /// </summary>
        /// <param name="iteration">The current iteration</param>
        /// <param name="cost">The current function cost</param>
        /// <param name="accuracy">The current network accuracy</param>
        internal TrainingProgressEventArgs(int iteration, float cost, float accuracy)
        {
            Iteration = iteration;
            Result = new DatasetEvaluationResult(cost, accuracy);
        }
    }
}
