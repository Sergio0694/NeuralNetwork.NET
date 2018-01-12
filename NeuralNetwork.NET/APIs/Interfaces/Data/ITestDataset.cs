using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.APIs.Interfaces.Data
{
    /// <summary>
    /// An interface for a dataset used to test a network being trained
    /// </summary>
    public interface ITestDataset : IDataset
    {
        /// <summary>
        /// Gets the callback used to report the training progress
        /// </summary>
        [CanBeNull]
        Action<TrainingProgressEventArgs> ProgressCallback { get; set; }
    }
}
