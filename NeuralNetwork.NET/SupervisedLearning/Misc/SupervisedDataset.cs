using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A simple struct that keeps a reference of a training set and its expected results
    /// </summary>
    public struct SupervisedDataset
    {
        /// <summary>
        /// Gets the current dataset
        /// </summary>
        [NotNull]
        public double[,] X { get; }

        /// <summary>
        /// Gets the expected results for the current dataset
        /// </summary>
        [NotNull]
        public double[,] Y { get; }

        /// <summary>
        /// Creates a new dataset wrapper batch with the given data
        /// </summary>
        /// <param name="x">The batch data</param>
        /// <param name="y">The batch expected results</param>
        public SupervisedDataset([NotNull] double[,] x, [NotNull] double[,] y)
        {
            if (x.GetLength(0) != y.GetLength(0)) throw new ArgumentException("The number of samples in the data and results must be the same");
            X = x;
            Y = y;
        }
    }
}