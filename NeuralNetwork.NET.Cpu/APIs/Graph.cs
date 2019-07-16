using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.Network.Nodes;
using NeuralNetworkDotNet.Network.Nodes.Nullary;
using NeuralNetworkDotNet.Network.Nodes.Unary;

namespace NeuralNetworkDotNet.APIs
{
    /// <summary>
    /// A <see langword="class"/> with primitives to build a computational graph
    /// </summary>
    public static class Graph
    {
        /// <summary>
        /// Creates a new <see cref="Node"/> instance for a linear input
        /// </summary>
        /// <param name="size">The input size</param>
        [Pure, NotNull]
        public static Node Linear(int size) => new PlaceholderNode((size, 1, 1));

        /// <summary>
        /// Creates a new <see cref="Node"/> instance for with a custom 3D shape
        /// </summary>
        /// <param name="height">The input volume height</param>
        /// <param name="width">The input volume width</param>
        /// <param name="channels">The number of channels in the input volume</param>
        [Pure, NotNull]
        public static Node Volume(int height, int width, int channels) => new PlaceholderNode((channels, height, width));

        /// <summary>
        /// Creates a new fully connected node with the specified number of output neurons
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="neurons">The number of output neurons in the new <see cref="Node"/></param>
        /// <returns>The resulting fully connected <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node FullyConnected([NotNull] this Node node, int neurons) => new FullyConnecteNode(node, neurons, WeightsInitializationMode.GlorotUniform, BiasInitializationMode.Zero);

        /// <summary>
        /// Creates a new fully connected node with the specified number of output neurons
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="neurons">The number of output neurons in the new <see cref="Node"/></param>
        /// <param name="weightsMode">The desired initialization mode to use for the weights of the node</param>
        /// <param name="biasMode">The desired initialization mode to use for the biases of the node</param>
        /// <returns>The resulting fully connected <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node FullyConnected([NotNull] this Node node, int neurons, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode) => new FullyConnecteNode(node, neurons, weightsMode, biasMode);

        /// <summary>
        /// Creates a new convolutional node with the specified number of kernels of a given size
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="size">The size of each convolutional kernel</param>
        /// <param name="kernels">The number of kernels to use in the node</param>
        /// <returns>The resulting convolutional <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node Convolution([NotNull] this Node node, (int X, int Y) size, int kernels) => new ConvolutionalNode(node, ConvolutionInfo.Default, size, kernels, BiasInitializationMode.Zero);

        /// <summary>
        /// Creates a new convolutional node with the specified number of kernels of a given size
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="size">The size of each convolutional kernel</param>
        /// <param name="kernels">The number of kernels to use in the node</param>
        /// <param name="biasMode">The desired initialization mode to use for the biases of the node</param>
        /// <returns>The resulting convolutional <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node Convolution([NotNull] this Node node, (int X, int Y) size, int kernels, BiasInitializationMode biasMode) => new ConvolutionalNode(node, ConvolutionInfo.Default, size, kernels, biasMode);

        /// <summary>
        /// Creates a pooling node with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <returns>The resulting pooling <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node Pooling([NotNull] this Node node) => new PoolingNode(node, PoolingInfo.Default);

        /// <summary>
        /// Creates an activation node with a specified activation function
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="type">The activation type to use in the new node</param>
        /// <returns>The resulting activation <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node Activation([NotNull] this Node node, ActivationType type) => new ActivationNode(node, type);

        /// <summary>
        /// Creates a new batch normalization node with the specified mode
        /// </summary>
        /// <param name="node">The source <see cref="Node"/> to connect to the new operation</param>
        /// <param name="mode">The normalization mode to use in the new node</param>
        /// <returns>The resulting batch normalization <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public static Node BatchNormalization([NotNull] this Node node, NormalizationMode mode) => new BatchNormalizationNode(node, mode);
    }
}
