using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.APIs.Interfaces.Data
{
    /// <summary>
    /// An interface for a dataset used to train or test a network
    /// </summary>
    [PublicAPI]
    public interface IDataset
    {
        /// <summary>
        /// Gets the number of samples in the current dataset
        /// </summary>
        int Count { get; }

        /// <summary>
        /// Gets the number of features in each dataset sample
        /// </summary>
        int InputFeatures { get; }

        /// <summary>
        /// Gets the number of features in each output sample in the dataset
        /// </summary>
        int OutputFeatures { get; }

        /// <summary>
        /// Gets the total raw size in bytes for the current dataset
        /// </summary>
        long ByteSize { get; }

        /// <summary>
        /// Gets the dataset sample at the input position. Note that the dataset is shuffled during training.
        /// </summary>
        /// <param name="i">The index of the sample to retrieve</param>
        DatasetSample this[int i] { get; }

        /// <summary>
        /// Gets a unique content id for the current dataset (not the same as <see cref="System.Object.GetHashCode"/> method
        /// that can be used to compare two <see cref="IDataset"/> instances and check if they contain the same samples (regardless of their relative order)
        /// </summary>
        int Id { get; }
    }
}