using System;
using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
        /// Artificially expands the current dataset by reinterpreting each sample as an image of the specified type and applying the input transformation to each sample to create new ones
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="width">The width of each sample image</param>
        /// <param name="height">The height of each sample image</param>
        /// <param name="factories">The list of functions to use to process the images and generate new samples</param>
        void Expand<TPixel>(int width, int height, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] factories) where TPixel : struct, IPixel<TPixel>;

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
