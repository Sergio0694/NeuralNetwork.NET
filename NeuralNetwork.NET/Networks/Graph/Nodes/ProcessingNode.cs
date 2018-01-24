using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class representing a single node in a computation graph
    /// </summary>
    internal sealed class ProcessingNode : NodeBase
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
        public IComputationGraphNode Parent { get; }

        internal ProcessingNode([NotNull] INetworkLayer layer, [NotNull] IComputationGraphNode parent) : base(ComputationGraphNodeType.Processing)
        {
            Layer = layer;
            Parent = parent;
        }

        /// <inheritdoc/>
        public override bool Equals(IComputationGraphNode other)
        {
            return base.Equals(other) &&
                   other is ProcessingNode processing &&
                   processing.Layer.Equals(Layer);
        }

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            Layer.To<INetworkLayer, NetworkLayerBase>().Serialize(stream);
        }
    }
}
