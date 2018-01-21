using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class representing a junction node in a computation graph
    /// </summary>
    public sealed class MergeNode : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; }

        /// <summary>
        /// Gets the list of parents nodes to merge for the current node
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<IComputationGraphNode> Parents { get; }

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children { get; }

        internal MergeNode(ComputationGraphNodeType type, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> parents, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> children)
        {
            Type = type == ComputationGraphNodeType.DepthConcatenation || type == ComputationGraphNodeType.Sum
                ? type
                : throw new ArgumentOutOfRangeException(nameof(type), "The graph node type is invalid for the current instance");
            Parents = parents.Count >= 2 ? parents : throw new ArgumentException("The number of parents must be at least equal to two", nameof(parents));
            Children = children.Count > 0 ? children : throw new ArgumentException("The number of child nodes must be greater than 0", nameof(children));
        }
    }
}
