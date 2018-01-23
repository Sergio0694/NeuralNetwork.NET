using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A base class for all the available node types
    /// </summary>
    internal abstract class NodeBase : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; }

        [CanBeNull, ItemNotNull]
        private IReadOnlyList<IComputationGraphNode> _Children;

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children
        {
            get => _Children ?? throw new InvalidOperationException("Node not initialized");
            set => _Children = _Children == null ? value : throw new InvalidOperationException("Node already initialized");
        }

        protected NodeBase(ComputationGraphNodeType type) => Type = type;

        /// <inheritdoc/>
        public virtual bool Equals(IComputationGraphNode other)
        {
            if (other == null) return false;
            if (other == this) return true;
            return other.GetType() == GetType() &&
                   other.Type == Type;
        }
    }
}
