using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.SupervisedLearning.Parameters
{
    /// <summary>
    /// A class that contains additional parameters to test a network being trained to monitor the general progress
    /// </summary>
    internal sealed class TestDataset : DatasetBase, ITestDataset
    {
        /// <inheritdoc/>
        public IProgress<TrainingProgressEventArgs> ProgressCallback { get; set; }

        public TestDataset((float[,] X, float[,] Y) testSet, [CanBeNull] IProgress<TrainingProgressEventArgs> callback) : base(testSet)
        {
            ProgressCallback = callback;
        }
    }
}
