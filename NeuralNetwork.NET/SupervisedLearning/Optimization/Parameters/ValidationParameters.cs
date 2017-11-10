using System;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    public sealed class ValidationParameters : DatasetParametersBase
    {
        // TODO: add docs
        public float Tolerance { get; }

        public int EpochsInterval { get; }

        public ValidationParameters((float[,] X, float[,] Y) validationSet, float tolerance, int epochs) : base(validationSet)
        {
            if (tolerance <= 0) throw new ArgumentOutOfRangeException(nameof(tolerance), "The tolerance must be a positive value");
            if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs), "The number of epochs must be at least equal to 1");
            Tolerance = tolerance;
            EpochsInterval = epochs;
        }
    }
}
