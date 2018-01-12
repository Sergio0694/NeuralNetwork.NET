using System;
using NeuralNetworkNET.APIs.Interfaces.Data;

namespace NeuralNetworkNET.SupervisedLearning.Parameters
{
    /// <summary>
    /// A class that contains additional parameters to validate the training data during a training session
    /// </summary>
    internal sealed class ValidationDataset : DatasetBase, IValidationDataset
    {
        /// <inheritdoc/>
        public float Tolerance { get; set; }

        /// <inheritdoc/>
        public int EpochsInterval { get; set; }

        public ValidationDataset((float[,] X, float[,] Y) validationSet, float tolerance, int epochs) : base(validationSet)
        {
            if (tolerance <= 0) throw new ArgumentOutOfRangeException(nameof(tolerance), "The tolerance must be a positive value");
            if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs), "The number of epochs must be at least equal to 1");
            Tolerance = tolerance;
            EpochsInterval = epochs;
        }
    }
}
