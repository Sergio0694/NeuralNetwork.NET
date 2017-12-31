using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Optimization.Progress;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    /// <summary>
    /// A class that contains additional parameters to test a network being trained to monitor the general progress
    /// </summary>
    public sealed class TestParameters : DatasetParametersBase
    {
        /// <summary>
        /// Gets the callback used to report the training progress
        /// </summary>
        [NotNull]
        public IProgress<BackpropagationProgressEventArgs> ProgressCallback { get; }

        public TestParameters((float[,] X, float[,] Y) testSet, [NotNull] IProgress<BackpropagationProgressEventArgs> callback) : base(testSet)
        {
            ProgressCallback = callback;
        }
    }
}
