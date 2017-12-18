using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes the available network layer types
    /// </summary>
    public static class NetworkLayers
    {
        /// <summary>
        /// Creates a new fully connected layer with the specified number of input and output neurons, and the given activation function
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(in TensorInfo input, int neurons, ActivationFunctionType activation) 
            => new FullyConnectedLayer(input, neurons, activation);

        /// <summary>
        /// Creates an output fully connected layer, with the specified cost function to use
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="cost">The cost function that should be used by the output layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(in TensorInfo input, int neurons, ActivationFunctionType activation, CostFunctionType cost) 
            => new OutputLayer(input, neurons, activation, cost);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="outputs">The number of output neurons</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Softmax(in TensorInfo input, int outputs) => new SoftmaxLayer(input, outputs);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Convolutional(in TensorInfo input, (int X, int Y) kernel, int kernels, ActivationFunctionType activation) 
            => new ConvolutionalLayer(input, kernel, kernels, activation);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Pooling(in TensorInfo input, ActivationFunctionType activation) => new PoolingLayer(input, activation);
    }
}
