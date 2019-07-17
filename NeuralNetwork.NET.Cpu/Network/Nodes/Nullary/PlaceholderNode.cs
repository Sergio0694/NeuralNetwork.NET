using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Nodes.Enums;

namespace NeuralNetworkDotNet.Network.Nodes.Nullary
{
    /// <summary>
    /// A placeholder node used for the inputs to a computational graph
    /// </summary>
    internal sealed class PlaceholderNode : Node
    {
        /// <inheritdoc/>
        public override NodeType Type => NodeType.Placeholder;

        public PlaceholderNode(Shape shape) : base(shape) { }
    }
}
