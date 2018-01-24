using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph.Nodes.Abstract
{
    /// <summary>
    /// A class representing a junction node in a computation graph
    /// </summary>
    internal abstract class MergeNodeBase : NodeBase
    {
        /// <summary>
        /// Gets the list of parents nodes to merge for the current node
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<IComputationGraphNode> Parents { get; }

        protected MergeNodeBase(ComputationGraphNodeType type, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(type)
        {
            Parents = parents.Count >= 2 ? parents : throw new ArgumentException("The number of parents must be at least equal to two", nameof(parents));
        }
    }
}
