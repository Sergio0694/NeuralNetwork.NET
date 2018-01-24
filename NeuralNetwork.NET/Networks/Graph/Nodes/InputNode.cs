using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class that represents the root node for a computation graph
    /// </summary>
    internal sealed class InputNode : NodeBase
    {
        public InputNode() : base(ComputationGraphNodeType.Input) { }
    }
}
