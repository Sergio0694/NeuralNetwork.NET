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
        /// Gets the number of inputs in the current layer
        /// </summary>
        int Inputs { get; }

        /// <summary>
        /// Gets the number of outputs in the current layer
        /// </summary>
        int Outputs { get; }
    }
}
