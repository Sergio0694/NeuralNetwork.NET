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
    internal sealed class MergeNode : NodeBase
    {
        /// <summary>
        /// Gets the list of parents nodes to merge for the current node
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<IComputationGraphNode> Parents { get; }

        internal MergeNode(ComputationGraphNodeType type, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(type)
        {
            if (type != ComputationGraphNodeType.DepthConcatenation && type != ComputationGraphNodeType.Sum)
                throw new ArgumentOutOfRangeException(nameof(type), "The graph node type is invalid for the current instance");
            Parents = parents.Count >= 2 ? parents : throw new ArgumentException("The number of parents must be at least equal to two", nameof(parents));
        }
    }
}
