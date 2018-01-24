using System;
using System.Collections.Generic;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Networks.Graph.Nodes.Abstract
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

        /// <summary>
        /// Writes the current node to the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/> to use to write the node data</param>
        public virtual void Serialize([NotNull] Stream stream) => stream.Write(Type);
    }
}
