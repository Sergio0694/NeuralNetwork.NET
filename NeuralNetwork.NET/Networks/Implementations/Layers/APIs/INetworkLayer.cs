using System;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.Networks.Implementations.Layers.APIs
{
    /// <summary>
    /// An interface that represents a single layer in a multilayer neural network
    /// </summary>
    public interface INetworkLayer : IEquatable<INetworkLayer>, IClonable<INetworkLayer>
    {
        /// <summary>
        /// Gets the number of inputs in the current layer
        /// </summary>
        int Inputs { get; }

        /// <summary>
        /// Gets the number of outputs in the current layer
        /// </summary>
        int Outputs { get; }

        /// <summary>
        /// Gets the activation type used in the current layer
        /// </summary>
        ActivationFunctionType ActivationFunctionType { get; }
    }
}
