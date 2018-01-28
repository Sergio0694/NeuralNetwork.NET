using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Networks.Layers.Cuda;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes the available cuDNN network layer types
    /// </summary>
    public static class CuDnnNetworkLayers
    {
        /// <summary>
        /// Gets whether or not the Cuda acceleration is supported on the current system
        /// </summary>
        public static bool IsCudaSupportAvailable => CuDnnService.IsAvailable;

        /// <summary>
        /// Creates a new fully connected layer with the specified number of input and output neurons, and the given activation function
        /// </summary>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory FullyConnected(
            int neurons, ActivationType activation,
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnFullyConnectedLayer(input, neurons, activation, weightsMode, biasMode);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="outputs">The number of output neurons</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Softmax(
            int outputs, 
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnSoftmaxLayer(input, outputs, weightsMode, biasMode);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Convolutional(
            (int X, int Y) kernel, int kernels, ActivationType activation,
            BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnConvolutionalLayer(input, ConvolutionInfo.Default, kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="info">The info on the convolution operation to perform</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Convolutional(
            ConvolutionInfo info, (int X, int Y) kernel, int kernels, ActivationType activation,
            BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnConvolutionalLayer(input, info, kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="factory">The <see cref="ConvolutionInfoFactory"/> instance to create a <see cref="ConvolutionInfo"/> value to use</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Convolutional(
            [NotNull] ConvolutionInfoFactory factory, (int X, int Y) kernel, int kernels, ActivationType activation,
            BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnConvolutionalLayer(input, factory(input, kernel), kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Pooling(ActivationType activation) => input => new CuDnnPoolingLayer(input, PoolingInfo.Default, activation);

        /// <summary>
        /// Creates a pooling layer with a custom mode, window size and stride
        /// </summary>
        /// <param name="info">The info on the pooling operation to perform</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Pooling(PoolingInfo info, ActivationType activation) => input => new CuDnnPoolingLayer(input, info, activation);

        /// <summary>
        /// Creates a new inception layer with the given features
        /// </summary>
        /// <param name="info">The info on the operations to execute inside the layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Inception(InceptionInfo info, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnInceptionLayer(input, info, biasMode);

        /// <summary>
        /// Creates a new batch normalization layer
        /// </summary>
        /// <param name="mode">The normalization mode to use for the new layer</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory BatchNormalization(NormalizationMode mode, ActivationType activation) 
            => input => new CuDnnBatchNormalizationLayer(input, mode, activation);
    }
}