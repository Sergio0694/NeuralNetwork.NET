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
        private Action<TrainingProgressEventArgs> _ProgressCallback;

        /// <inheritdoc/>
        public Action<TrainingProgressEventArgs> ProgressCallback
        {
            get => _ProgressCallback;
            set
            {
                _ProgressCallback = value;
                ThreadSafeProgressCallback = value == null 
                    ? null 
                    : new Progress<TrainingProgressEventArgs>(value); // Creating a Progress instance captures the current synchronization context
            }
        }
        
        /// <summary>
        /// Gets the <see cref="IProgress{T}"/> instance to safely report the progress from any source thread
        /// </summary>
        internal IProgress<TrainingProgressEventArgs> ThreadSafeProgressCallback { get; private set; } 

        public TestDataset((float[,] X, float[,] Y) testSet, [CanBeNull] Action<TrainingProgressEventArgs> callback) : base(testSet) => ProgressCallback = callback;
    }
}
