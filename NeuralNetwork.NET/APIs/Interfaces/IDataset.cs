using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface for a dataset used to train or test a network
    /// </summary>
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
        /// Gets or sets the number of samples in each samples batch in the current dataset
        /// </summary>
        int BatchSize { get; set; }

        /// <summary>
        /// Gets the total raw size in bytes for the current dataset
        /// </summary>
        long ByteSize { get; }

        /// <summary>
        /// Gets the dataset sample at the input position. Note that the dataset is shuffled during training.
        /// </summary>
        /// <param name="i">The index of the sample to retrieve</param>
        DatasetSample this[int i] { get; }
    }
}