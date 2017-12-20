using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
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
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer FullyConnected(
            in TensorInfo input, int neurons, ActivationFunctionType activation,
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => new CuDnnFullyConnectedLayer(input, neurons, activation, weightsMode, biasMode);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> descriptor</param>
        /// <param name="outputs">The number of output neurons</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Softmax(
            in TensorInfo input, int outputs, 
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => new CuDnnSoftmaxLayer(input, outputs, weightsMode, biasMode);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="input">The input volume to process</param>
        /// <param name="info">The info on the convolution operation to perform</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Convolutional(
            in TensorInfo input, 
            in ConvolutionInfo info, (int X, int Y) kernel, int kernels, ActivationFunctionType activation,
            BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => new CuDnnConvolutionalLayer(input, info, kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="input">The input volume to pool</param>
        /// <param name="info">The info on the pooling operation to perform</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INetworkLayer Pooling(in TensorInfo input, in PoolingInfo info, ActivationFunctionType activation) => new CuDnnPoolingLayer(input, info, activation);
    }
}