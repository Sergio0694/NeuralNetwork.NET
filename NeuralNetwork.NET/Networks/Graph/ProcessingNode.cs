using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class representing a single node in a computation graph
    /// </summary>
    public sealed class ProcessingNode : IComputationGraphNode
    {
        /// <inheritdoc/>
        public ComputationGraphNodeType Type { get; } = ComputationGraphNodeType.Processing;

        /// <summary>
        /// Gets the layer associated with the current graph node
        /// </summary>
        [NotNull]
        public INetworkLayer Layer { get; }
        
        /// <summary>
        /// Gets the parent node for the current graph node
        /// </summary>
        [NotNull]
        public IComputationGraphNode Parent { get; }

        /// <inheritdoc/>
        public IReadOnlyList<IComputationGraphNode> Children { get; }

        internal ProcessingNode([NotNull] INetworkLayer layer, [NotNull] IComputationGraphNode parent, [NotNull, ItemNotNull] IReadOnlyList<IComputationGraphNode> children)
        {
            Layer = layer;
            Parent = parent;
            Children = children;
        }
    }
}
