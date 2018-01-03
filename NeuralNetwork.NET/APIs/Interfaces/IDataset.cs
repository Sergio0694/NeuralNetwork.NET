using System;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface for a dataset used to train or test a network
    /// </summary>
    public interface IDataset
    {
        /// <summary>
        /// Gets the total number of samples in the current dataset
        /// </summary>
        int SamplesCount { get; }

        /// <summary>
        /// Gets the i-th sample in the dataset
        /// </summary>
        /// <param name="i">The index of the sample to retrieve</param>
        (Span<float> X, Span<float> Y) this[int i] { get; }
    }
}