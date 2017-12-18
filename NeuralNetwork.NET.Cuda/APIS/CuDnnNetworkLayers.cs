using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes the available cuDNN network layer types
    /// </summary>
    public static class CuDnnNetworkLayers
    {
        /// <summary>
        /// Creates a new fully connected layer with the specified number of input and output neurons, and the given activation function
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(in TensorInfo input, int neurons, ActivationFunctionType activation) => new CuDnnFullyConnectedLayer(input, neurons, activation);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="outputs">The number of output neurons</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Softmax(in TensorInfo input, int outputs) => new CuDnnSoftmaxLayer(input, outputs);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="input">The input volume to process</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="mode">The desired convolution mode to use</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Convolutional(TensorInfo input, (int X, int Y) kernel, int kernels, ActivationFunctionType activation, ConvolutionMode mode = ConvolutionMode.CONVOLUTION) => new CuDnnConvolutionalLayer(input, kernel, kernels, activation, mode);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="input">The input volume to pool</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Pooling(TensorInfo input, ActivationFunctionType activation) => new CuDnnPoolingLayer(input, activation);
    }
}