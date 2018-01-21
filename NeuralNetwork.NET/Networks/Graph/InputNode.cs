using NeuralNetworkNET.APIs.Enums;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A class that represents the root node for a computation graph
    /// </summary>
    internal sealed class InputNode : NodeBase
    {
        public InputNode() : base(ComputationGraphNodeType.Input) { }
    }
}
