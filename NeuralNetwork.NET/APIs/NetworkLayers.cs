using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Layers.Cpu;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes the available network layer types
    /// </summary>
    public static class NetworkLayers
    {
        /// <summary>
        /// Creates a new fully connected layer with the specified number of output neurons, and the given activation function
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
            => input => new FullyConnectedLayer(input, neurons, activation, weightsMode, biasMode);

        /// <summary>
        /// Creates an output fully connected layer, with the specified cost function to use
        /// </summary>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="cost">The cost function that should be used by the output layer</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory FullyConnected(
            int neurons, ActivationType activation, CostFunctionType cost,
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new OutputLayer(input, neurons, activation, cost, weightsMode, biasMode);

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
            => input => new SoftmaxLayer(input, outputs, weightsMode, biasMode);

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
            => input => new ConvolutionalLayer(input, ConvolutionInfo.Default, kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Pooling(ActivationType activation) => input => new PoolingLayer(input, PoolingInfo.Default, activation);

        /// <summary>
        /// Creates a new batch normalization layer
        /// </summary>
        /// <param name="mode">The normalization mode to use for the new layer</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory BatchNormalization(NormalizationMode mode, ActivationType activation)
            => input => new BatchNormalizationLayer(input, mode, activation);
    }
}
