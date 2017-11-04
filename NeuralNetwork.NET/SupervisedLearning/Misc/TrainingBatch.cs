using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A simple struct that contains a reference to the data in a training batch
    /// </summary>
    internal struct TrainingBatch
    {
        /// <summary>
        /// Gets the training data for the current batch
        /// </summary>
        [NotNull]
        public double[,] X { get; }

        /// <summary>
        /// Gets the expected results for the current batch
        /// </summary>
        [NotNull]
        public double[,] Y { get; }

        /// <summary>
        /// Creates a new training batch with the given data
        /// </summary>
        /// <param name="x">The batch data</param>
        /// <param name="y">The batch expected results</param>
        public TrainingBatch([NotNull] double[,] x, [NotNull] double[,] y)
        {
            if (x.GetLength(0) != y.GetLength(0)) throw new ArgumentException("The number of samples in the batch data and results must be the same");
            X = x;
            Y = y;
        }
    }
}