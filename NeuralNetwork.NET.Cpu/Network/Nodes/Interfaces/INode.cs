using NeuralNetworkDotNet.APIs.Structs;

namespace NeuralNetworkDotNet.Network.Nodes.Interfaces
{
    /// <summary>
    /// An <see langword="interface"/> for all the available nodes
    /// </summary>
    public interface INode
    {
        /// <summary>
        /// Gets the shape of the node outputs
        /// </summary>
        Shape Shape { get; }
    }
}
