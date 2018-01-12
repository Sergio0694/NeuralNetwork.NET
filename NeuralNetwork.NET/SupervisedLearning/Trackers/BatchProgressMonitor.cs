using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.SupervisedLearning.Trackers
{
    /// <summary>
    /// A simple monitor object that keeps track of the current training progress and notifies every time a batch is completed
    /// </summary>
    internal sealed class BatchProgressMonitor
    {
        // The total number of items in the training dataset
        private readonly int WorkItems;

        // The progress callback to notify when a new batch is completed
        [NotNull]
        private readonly IProgress<BatchProgress> Callback;

        // Internal constructor for the network trainer
        public BatchProgressMonitor(int items, [NotNull] IProgress<BatchProgress> callback)
        {
            WorkItems = items;
            Callback = callback;
        }

        // The current number of processed samples from the training dataset
        private int _ProcessedItems;

        /// <summary>
        /// Notifies the completion of a new training batch
        /// </summary>
        /// <param name="size">The size of the completed training batch</param>
        public void NotifyCompletedBatch(int size)
        {
            _ProcessedItems += size;
            Callback.Report(new BatchProgress(_ProcessedItems, _ProcessedItems * 100f / WorkItems));
        }

        /// <summary>
        /// Resets the progress monitor at the end of a training epoch
        /// </summary>
        public void Reset() => _ProcessedItems = 0;
    }
}
