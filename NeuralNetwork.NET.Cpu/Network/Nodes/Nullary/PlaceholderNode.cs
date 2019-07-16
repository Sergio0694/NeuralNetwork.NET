using NeuralNetworkDotNet.APIs.Structs;

namespace NeuralNetworkDotNet.Network.Nodes.Nullary
{
    /// <summary>
    /// A placeholder node used for the inputs to a computational graph
    /// </summary>
    internal sealed class PlaceholderNode : Node
    {
        public PlaceholderNode(Shape shape) : base(shape) { }
    }
}
