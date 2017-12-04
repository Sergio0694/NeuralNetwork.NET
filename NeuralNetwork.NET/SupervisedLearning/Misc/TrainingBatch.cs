using System;
using System.Diagnostics;
using JetBrains.Annotations;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A simple struct that keeps a reference of a training set and its expected results
    /// </summary>
    [DebuggerDisplay("Samples: {X.GetLength(0)}, inputs: {X.GetLength(1)}, outputs: {Y.GetLength(1)}")]
    public readonly struct TrainingBatch
    {
        /// <summary>
        /// Gets the current dataset
        /// </summary>
        [NotNull]
        public readonly float[,] X;

        /// <summary>
        /// Gets the expected results for the current dataset
        /// </summary>
        [NotNull]
        public readonly float[,] Y;

        /// <summary>
        /// Creates a new dataset wrapper batch with the given data
        /// </summary>
        /// <param name="x">The batch data</param>
        /// <param name="y">The batch expected results</param>
        public TrainingBatch([NotNull] float[,] x, [NotNull] float[,] y)
        {
            if (x.GetLength(0) != y.GetLength(0)) throw new ArgumentException("The number of samples in the data and results must be the same");
            X = x;
            Y = y;
        }
    }
}