using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;

namespace NeuralNetworkNET.Networks.Implementations.Layers.APIs
{
    /// <summary>
    /// A static class that exposes the available network layer types
    /// </summary>
    public static class NetworkLayers
    {
        /// <summary>
        /// Creates a new fully connected layer with the specified number of input and output neurons, and the given activation function
        /// </summary>
        /// <param name="inputs">The number of input neurons</param>
        /// <param name="outputs">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(int inputs, int outputs, ActivationFunctionType activation) => new FullyConnectedLayer(inputs, outputs, activation);

        /// <summary>
        /// Creates an output fully connected layer, with the specified cost function to use
        /// </summary>
        /// <param name="inputs">The number of input neurons</param>
        /// <param name="outputs">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="cost">The cost function that should be used by the output layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost) => new OutputLayer(inputs, outputs, activation, cost);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="inputs">The number of input neurons</param>
        /// <param name="outputs">The number of output neurons</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Softmax(int inputs, int outputs) => new SoftmaxLayer(inputs, outputs);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="input">The input volume to process</param>
        /// <param name="kernelAxis">The size of each 2D axis of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Convolutional(VolumeInformation input, int kernelAxis, int kernels, ActivationFunctionType activation) => new ConvolutionalLayer(input, kernelAxis, kernels, activation);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="input">The input volume to pool</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Pooling(VolumeInformation input, ActivationFunctionType activation) => new PoolingLayer(input, activation);
    }
}
