using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    public sealed class TestParameters : DatasetParametersBase
    {
        public IProgress<BackpropagationProgressEventArgs> ProgressCallback { get; }

        public TestParameters((float[,] X, float[,] Y) testSet, [NotNull] IProgress<BackpropagationProgressEventArgs> callback) : base(testSet)
        {
            ProgressCallback = callback;
        }
    }
}
