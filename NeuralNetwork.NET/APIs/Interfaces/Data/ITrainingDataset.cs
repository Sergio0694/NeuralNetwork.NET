using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.APIs.Interfaces.Data
{
    /// <summary>
    /// An interface for a batched dataset used to train a network
    /// </summary>
    public interface ITrainingDataset : IDataset
    {
        /// <summary>
        /// Gets or sets the number of samples in each samples batch in the current dataset
        /// </summary>
        int BatchSize { get; set; }

        /// <summary>
        /// Gets the number of training batches in the current dataset (according to the number of samples and the batch size)
        /// </summary>
        int BatchesCount { get; }

        /// <summary>
        /// Artificially expands the current dataset applying the input transformation to each sample to create new ones
        /// </summary>
        /// <param name="factories">The list of functions to use to generate new samples from each one in the dataset</param>
        void Expand([NotNull, ItemNotNull] params Func<float[], float[]>[] factories);

        /// <summary>
        /// Removes a specified fraction of samples from the current instance and returns a new <see cref="ITestDataset"/>
        /// </summary>
        /// <param name="ratio">The ratio of samples to include in the returned <see cref="ITrainingDataset"/></param>
        /// <param name="progress">The optional progress callback to use</param>
        [Pure, NotNull]
        ITestDataset ExtractTest(float ratio, [CanBeNull] Action<TrainingProgressEventArgs> progress = null);

        /// <summary>
        /// Returns a pair of new datasets, where the first is an <see cref="ITrainingDataset"/> with the specified fraction of samples and 
        /// the second is an <see cref="IValidationDataset"/> with the remaining number of samples from the current dataset
        /// </summary>
        /// <param name="ratio">The ratio of samples to include in the returned <see cref="ITrainingDataset"/></param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        [Pure, NotNull]
        IValidationDataset ExtractValidation(float ratio, float tolerance = 1e-2f, int epochs = 5);

        /// <summary>
        /// Returns a pair of new datasets, where the first is an <see cref="ITrainingDataset"/> with the specified fraction of samples and 
        /// the second is an <see cref="ITestDataset"/> with the remaining number of samples from the current dataset
        /// </summary>
        /// <param name="ratio">The ratio of samples to include in the returned <see cref="ITrainingDataset"/></param>
        /// <param name="progress">The optional progress callback to use</param>
        [Pure]
        (ITrainingDataset, ITestDataset) PartitionWithTest(float ratio, [CanBeNull] Action<TrainingProgressEventArgs> progress = null);

        /// <summary>
        /// Returns a pair of new datasets, where the first is an <see cref="ITrainingDataset"/> with the specified fraction of samples and 
        /// the second is an <see cref="IValidationDataset"/> with the remaining number of samples from the current dataset
        /// </summary>
        /// <param name="ratio">The ratio of samples to include in the returned <see cref="ITrainingDataset"/></param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        [Pure]
        (ITrainingDataset, IValidationDataset) PartitionWithValidation(float ratio, float tolerance = 1e-2f, int epochs = 5);
    }
}
