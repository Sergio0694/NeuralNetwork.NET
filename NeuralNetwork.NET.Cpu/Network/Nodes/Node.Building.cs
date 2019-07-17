using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Graph;
using NeuralNetworkDotNet.Network.Nodes.Abstract;
using NeuralNetworkDotNet.Network.Nodes.Binary;
using NeuralNetworkDotNet.Network.Nodes.Nullary;
using NeuralNetworkDotNet.Network.Nodes.Unary;
using NeuralNetworkDotNet.Network.Nodes.Unary.Losses;

namespace NeuralNetworkDotNet.Network.Nodes
{
    public abstract partial class Node
    {
        /// <summary>
        /// Creates a new fully connected node with the specified number of output neurons
        /// </summary>
        /// <param name="neurons">The number of output neurons in the new <see cref="Node"/></param>
        /// <returns>The resulting fully connected <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node FullyConnected(int neurons) => new FullyConnectedNode(this, neurons, WeightsInitializationMode.GlorotUniform, BiasInitializationMode.Zero);

        /// <summary>
        /// Creates a new fully connected node with the specified number of output neurons
        /// </summary>
        /// <param name="neurons">The number of output neurons in the new <see cref="Node"/></param>
        /// <param name="weightsMode">The desired initialization mode to use for the weights of the node</param>
        /// <param name="biasMode">The desired initialization mode to use for the biases of the node</param>
        /// <returns>The resulting fully connected <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node FullyConnected(int neurons, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode) => new FullyConnectedNode(this, neurons, weightsMode, biasMode);

        /// <summary>
        /// Creates a new convolutional node with the specified number of kernels of a given size
        /// </summary>
        /// <param name="size">The size of each convolutional kernel</param>
        /// <param name="kernels">The number of kernels to use in the node</param>
        /// <returns>The resulting convolutional <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Convolution((int X, int Y) size, int kernels) => new ConvolutionalNode(this, ConvolutionInfo.Default, size, kernels, BiasInitializationMode.Zero);

        /// <summary>
        /// Creates a new convolutional node with the specified number of kernels of a given size
        /// </summary>
        /// <param name="size">The size of each convolutional kernel</param>
        /// <param name="kernels">The number of kernels to use in the node</param>
        /// <param name="biasMode">The desired initialization mode to use for the biases of the node</param>
        /// <returns>The resulting convolutional <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Convolution((int X, int Y) size, int kernels, BiasInitializationMode biasMode) => new ConvolutionalNode(this, ConvolutionInfo.Default, size, kernels, biasMode);

        /// <summary>
        /// Creates a pooling node with a window of size 2 and a stride of 2
        /// </summary>
        /// <returns>The resulting pooling <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Pooling() => new PoolingNode(this, PoolingInfo.Default);

        /// <summary>
        /// Creates an activation node with a specified activation function
        /// </summary>
        /// <param name="type">The activation type to use in the new node</param>
        /// <returns>The resulting activation <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Activation(ActivationType type) => new ActivationNode(this, type);

        /// <summary>
        /// Creates a new batch normalization node with the specified mode
        /// </summary>
        /// <param name="mode">The normalization mode to use in the new node</param>
        /// <returns>The resulting batch normalization <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node BatchNormalization(NormalizationMode mode) => new BatchNormalizationNode(this, mode);

        /// <summary>
        /// Creates a new dropout node
        /// </summary>
        /// <returns>The resulting dropout <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Dropout() => new DropoutNode(this);

        /// <summary>
        /// Creates a new loss node with a specified activation function and loss function
        /// </summary>
        /// <param name="activation"></param>
        /// <param name="cost"></param>
        /// <returns>The resulting loss <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Loss(ActivationType activation, CostFunctionType cost) => new OutputNode(this, activation, cost);

        /// <summary>
        /// Creates a new softmax node
        /// </summary>
        /// <returns>The resulting softmax <see cref="Node"/> instance</returns>
        [Pure, NotNull]
        public Node Softmax() => new SoftmaxNode(this);

        /// <summary>
        /// Creates a new <see cref="APIs.Graph"/> instance from the current node
        /// </summary>
        /// <returns>The resulting <see cref="APIs.Graph"/> instance with the created nodes</returns>
        [Pure, NotNull]
        public INetwork Build()
        {
            Guard.IsTrue(this is OutputNode, "The last node must be an output node");

            var nodes = new HashSet<Node>();

            // Explore the nodes in the graph and extract all of them in a collection
            void ExploreNodes(Node current)
            {
                if (nodes.Contains(current)) return;
                nodes.Add(current);

                switch (current)
                {
                    case PlaceholderNode _: return;
                    case UnaryNodeBase unary:
                        ExploreNodes(unary.Parent);
                        break;
                    case BinaryNodeBase binary:
                        ExploreNodes(binary.LeftParent);
                        ExploreNodes(binary.RightParent);
                        break;
                    default: throw new InvalidOperationException($"Invalid node of type {current.GetType().FullName}");
                }
            }

            ExploreNodes(this);

            Guard.IsFalse(nodes.OfType<PlaceholderNode>().Count() > 1, "A graph can't contain more than a placeholder node");
            Guard.IsFalse(nodes.OfType<OutputNode>().Count() > 1, "A graph can't contain more than an output node");

            return new ComputationalGraph(nodes);
        }

        /// <summary>
        /// Sums the two input nodes and returns a new <see cref="Node"/> instance
        /// </summary>
        /// <param name="a">The first <see cref="Node"/> to sum</param>
        /// <param name="b">The second <see cref="Node"/> to sum</param>
        /// <returns>A new <see cref="Node"/> that performs the sum of the two input nodes</returns>
        [Pure, NotNull]
        public static Node operator +([NotNull] Node a, [NotNull] Node b) => new SumNode(a, b);

        /// <summary>
        /// Stacks (depth concatenation) the two input nodes and returns a new <see cref="Node"/> instance
        /// </summary>
        /// <param name="a">The first <see cref="Node"/> to stack</param>
        /// <param name="b">The second <see cref="Node"/> to stack</param>
        /// <returns>A new <see cref="Node"/> that performs the stack of the two input nodes</returns>
        [Pure, NotNull]
        public static Node operator |([NotNull] Node a, [NotNull] Node b) => new DepthConcatenationNode(a, b);
    }
}
