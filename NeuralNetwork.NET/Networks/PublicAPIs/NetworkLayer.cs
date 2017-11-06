using System;
using NeuralNetworkNET.Networks.Architecture;

namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// A class that represents a single layer in a multilayer neural network
    /// </summary>
    public class NetworkLayer
    {
        /// <summary>
        /// Gets the number of neurons in the current layer
        /// </summary>
        public int Neurons { get; }

        // Private base constructor
        private NetworkLayer(int neurons)
        {
            if (neurons <= 0) throw new ArgumentOutOfRangeException(nameof(neurons), "The number of neurons must be positive");
            Neurons = neurons;
        }

        /// <summary>
        /// Creates a new input network layer with the specified number of neurons
        /// </summary>
        /// <param name="neurons">The number of neurons in the input layer</param>
        public static NetworkLayer Inputs(int neurons) => new InputLayer(neurons);

        /// <summary>
        /// Creates a new fully connected layer with the specified number of neurons and activation function
        /// </summary>
        /// <param name="neurons">The number of neurons in the fully connected layer</param>
        /// <param name="activation">The activation funtion to use in the current layer</param>
        public static NetworkLayer FullyConnected(int neurons, ActivationFunction activation) => new FullyConnectedLayer(neurons, activation);

        /// <summary>
        /// An internal class representing the input layer of a neural network
        /// </summary>
        internal sealed class InputLayer : NetworkLayer
        {
            public InputLayer(int neurons) : base(neurons) { }
        }

        /// <summary>
        /// An internal class representing a fully connected layer with a specific activation function
        /// </summary>
        internal sealed class FullyConnectedLayer : NetworkLayer
        {
            /// <summary>
            /// Gets the activation function for the current layer
            /// </summary>
            public ActivationFunction Activation { get; }

            public FullyConnectedLayer(int neurons, ActivationFunction activation) : base(neurons)
            {
                Activation = activation;
            }
        }
    }
}
