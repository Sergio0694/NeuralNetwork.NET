using System;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    public sealed class ValidationParameters : DatasetParametersBase
    {
        // TODO: add docs
        public double Tolerance { get; }

        public int EpochsInterval { get; }

        public ValidationParameters((double[,] X, double[,] Y) validationSet, double tolerance, int epochs) : base(validationSet)
        {
            if (tolerance <= 0) throw new ArgumentOutOfRangeException(nameof(tolerance), "The tolerance must be a positive value");
            if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs), "The number of epochs must be at least equal to 1");
            Tolerance = tolerance;
            EpochsInterval = epochs;
        }
    }
}
