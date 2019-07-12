using System;

namespace NeuralNetworkDotNet.Core.APIs.Interfaces
{
    /// <summary>
    /// An interface that represents a single layer in a network
    /// </summary>
    public interface ILayer : IEquatable<ILayer>, IClonable<ILayer>
    {
        /// <summary>
        /// Gets the shape of the layer inputs
        /// </summary>
        (int C, int H, int W) InputShape { get; }

        /// <summary>
        /// Gets the shape of the layer outputs
        /// </summary>
        (int C, int H, int W) OutputShape { get; }
    }
}
