using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class representing a single node in a computation graph
    /// </summary>
    public sealed class ComputationGraphNode
    {
        /// <summary>
        /// Gets the layer associated with the current graph node
        /// </summary>
        [NotNull]
        public INetworkLayer Layer { get; }
        
        /// <summary>
        /// Gets the parent node for the current graph node
        /// </summary>
        [NotNull]
        public ComputationGraphNode Parent { get; }

        /// <summary>
        /// Gets the collection of children nodes for the current node
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<ComputationGraphNode> Children { get; }

        internal ComputationGraphNode([NotNull] INetworkLayer layer, [NotNull] ComputationGraphNode parent, [NotNull, ItemNotNull] IReadOnlyList<ComputationGraphNode> children)
        {
            Layer = layer;
            Parent = parent;
            Children = children;
        }
    }
}
