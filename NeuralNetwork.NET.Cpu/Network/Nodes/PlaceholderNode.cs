using NeuralNetworkDotNet.APIs.Structs;

namespace NeuralNetworkDotNet.Network.Nodes
{
    /// <summary>
    /// A placeholder node used for the inputs to a computational graph
    /// </summary>
    internal sealed class PlaceholderNode : INode
    {
        /// <inheritdoc/>
        public Shape Shape { get; }

        public PlaceholderNode(Shape shape) => Shape = shape;

        /// <inheritdoc/>
        public bool Equals(INode other) => other is PlaceholderNode node && Shape == node.Shape;
    }
}
