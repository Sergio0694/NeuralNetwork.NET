using System;
using NeuralNetworkDotNet.APIs.Structs;

namespace NeuralNetworkDotNet.APIs.Interfaces
{
    /// <summary>
    /// An interface that represents a single layer in a network
    /// </summary>
    public interface ILayer : IEquatable<ILayer>, IClonable<ILayer>
    {
        /// <summary>
        /// Gets the shape of the layer inputs
        /// </summary>
        Shape InputShape { get; }

        /// <summary>
        /// Gets the shape of the layer outputs
        /// </summary>
        Shape OutputShape { get; }
    }
}
