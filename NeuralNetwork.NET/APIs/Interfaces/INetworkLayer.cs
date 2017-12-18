using NeuralNetworkNET.APIs.Misc;
using System;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface that represents a single layer in a multilayer neural network
    /// </summary>
    public interface INetworkLayer : IEquatable<INetworkLayer>, IClonable<INetworkLayer>
    {
        /// <summary>
        /// Gets the kind of neural network layer
        /// </summary>
        LayerType LayerType { get; }

        /// <summary>
        /// Gets the info on the layer inputs
        /// </summary>
        TensorInfo InputInfo { get; }

        /// <summary>
        /// Gets the info on the layer outputs
        /// </summary>
        TensorInfo OutputInfo { get; }
    }
}
