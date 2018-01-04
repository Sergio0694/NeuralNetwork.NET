using System;
using System.Collections.Generic;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface for a dataset used to train or test a network
    /// </summary>
    public interface IDataset : IReadOnlyList<(Span<float> X, Span<float> Y)>
    {
        /// <summary>
        /// Gets the total raw size in bytes for the current dataset
        /// </summary>
        long ByteSize { get; }
    }
}