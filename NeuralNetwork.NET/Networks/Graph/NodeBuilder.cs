using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class that represents a declaration for a graph node to build
    /// </summary>
    public sealed class NodeBuilder
    {
        #region Fields and initializaation

        // The optional parameters for the node to build
        [CanBeNull]
        private object Parameter { get; }

        // The instance node type
        internal ComputationGraphNodeType NodeType { get; }

        /// <summary>
        /// Gets the current list of parent <see cref="NodeBuilder"/> instances
        /// </summary>
        [NotNull, ItemNotNull]
        internal List<NodeBuilder> Parents { get; } = new List<NodeBuilder>();

        /// <summary>
        /// Gets the current list of child <see cref="NodeBuilder"/> instances
        /// </summary>
        [NotNull, ItemNotNull]
        internal List<NodeBuilder> Children { get; } = new List<NodeBuilder>();

        internal NodeBuilder(ComputationGraphNodeType type, [CanBeNull] object parameter)
        {
            if (type == ComputationGraphNodeType.Processing && !(parameter is LayerFactory))
                throw new InvalidOperationException("Invalid node initialization");
            NodeType = type;
            Parameter = parameter;
        }

        // Static constructor for a node with multiple parents
        private NodeBuilder New(ComputationGraphNodeType type, [CanBeNull] object parameter, [NotNull, ItemNotNull] params NodeBuilder[] inputs)
        {
            if (inputs.Length < 1) throw new ArgumentException("The inputs must be at least two", nameof(inputs));
            NodeBuilder next = new NodeBuilder(type, parameter);
            Children.Add(next);
            foreach (NodeBuilder input in inputs)
                input.Children.Add(next);
            next.Parents.Add(this);
            next.Parents.AddRange(inputs);
            return next;
        }

        // Constructor for a node with a single parent
        private NodeBuilder New(ComputationGraphNodeType type, [CanBeNull] object parameter)
        {
            NodeBuilder next = new NodeBuilder(type, parameter);
            Children.Add(next);
            next.Parents.Add(this);
            return next;
        }

        /// <summary>
        /// Creates an input node to use to build a new graph
        /// </summary>
        [Pure, NotNull]
        internal static NodeBuilder Input() => new NodeBuilder(ComputationGraphNodeType.Input, null);

        /// <summary>
        /// Extracts the node parameter
        /// </summary>
        /// <typeparam name="T">The target parameter type</typeparam>
        [Pure]
        internal T GetParameter<T>() => Parameter != null && Parameter.GetType() == typeof(T)
            ? Parameter.To<object, T>()
            : throw new InvalidOperationException("Invalid parameter type");

        #endregion

        #region APIs

        /// <summary>
        /// Creates a new linear sum node that merges two input nodes
        /// </summary>
        /// <param name="first">The first node to sum</param>
        /// <param name="second">The second node to sum</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public static NodeBuilder operator +([NotNull] NodeBuilder first, [NotNull] NodeBuilder second) => first.Sum(second);

        /// <summary>
        /// Creates a new linear sum node that merges multiple input nodes
        /// </summary>
        /// <param name="inputs">The sequence of parent nodes for the new instance</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder Sum(params NodeBuilder[] inputs) => Sum(ActivationType.Identity, inputs);

        /// <summary>
        /// Creates a new linear sum node that merges multiple input nodes
        /// </summary>
        /// <param name="activation">The activation function to use in the sum layer</param>
        /// <param name="inputs">The sequence of parent nodes for the new instance</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder Sum(ActivationType activation, params NodeBuilder[] inputs)
        {
            ExecutionModePreference mode = CuDnnNetworkLayers.IsCudaSupportAvailable
                ? ExecutionModePreference.Cuda
                : ExecutionModePreference.Cpu;
            return New(ComputationGraphNodeType.Sum, (activation, mode), inputs);
        }

        /// <summary>
        /// Creates a new linear sum node that merges multiple input nodes
        /// </summary>
        /// <param name="activation">The activation function to use in the sum layer</param>
        /// <param name="mode">The desired execution preference for the activation function</param>
        /// <param name="inputs">The sequence of parent nodes for the new instance</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder Sum(ActivationType activation, ExecutionModePreference mode, params NodeBuilder[] inputs) => New(ComputationGraphNodeType.Sum, (activation, mode), inputs);

        /// <summary>
        /// Creates a new depth concatenation node that merges multiple input nodes with the same output shape
        /// </summary>
        /// <param name="inputs">The sequence of parent nodes for the new instance</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder DepthConcatenation(params NodeBuilder[] inputs) => New(ComputationGraphNodeType.DepthConcatenation, null, inputs);

        /// <summary>
        /// Creates a new node that will host a network layer to process its inputs
        /// </summary>
        /// <param name="factory">The <see cref="LayerFactory"/> instance used to create the network layer</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder Layer(LayerFactory factory) => New(ComputationGraphNodeType.Processing, factory);

        /// <summary>
        /// Creates a pipeline that contains a sequence of network layers
        /// </summary>
        /// <param name="factories">The list of <see cref="LayerFactory"/> instances to use to create the pipeline nodes</param>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder Pipeline([NotNull, ItemNotNull] params LayerFactory[] factories)
        {
            return factories.Length >= 2 
                ? factories.Aggregate(this, (s, f) => s.Layer(f))
                : throw new ArgumentException("A pipeline must contain at leaast two layers", nameof(factories));
        }

        /// <summary>
        /// Creates a new node that will route its inputs to a training sug-graph
        /// </summary>
        [PublicAPI]
        [MustUseReturnValue, NotNull]
        public NodeBuilder TrainingBranch() => New(ComputationGraphNodeType.TrainingBranch, null);

        #endregion
    }
}
