using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    public sealed class TestParameters : DatasetParametersBase
    {
        public IProgress<BackpropagationProgressEventArgs> ProgressCallback { get; }

        public TestParameters((double[,] X, double[,] Y) testSet, [NotNull] IProgress<BackpropagationProgressEventArgs> callback) : base(testSet)
        {
            ProgressCallback = callback;
        }
    }
}
