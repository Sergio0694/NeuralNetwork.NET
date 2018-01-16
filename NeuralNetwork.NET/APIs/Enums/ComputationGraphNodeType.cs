namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see langword="enum"/> that indicates the type of a node in a graph network
    /// </summary>
    public enum ComputationGraphNodeType : byte
    {
        /// <summary>
        /// A computation graph node with an associated <see cref="Interfaces.INetworkLayer"/> that processes an input <see cref="Structs.Tensor"/>
        /// </summary>
        Processing,

        /// <summary>
        /// A computation graph node that forwards its input to two different branches: an inference branch that eventually leads to the output
        /// layers using to evaluate the network, and a secondary training branch used during backpropagation to inject partial gradients
        /// </summary>
        TrainingSplit,

        /// <summary>
        /// A computation graph node that merges a series of inputs by stacking them along the depth axis
        /// </summary>
        DepthStacking,

        /// <summary>
        /// A computation graph node that merges a series of inputs by summing their values together
        /// </summary>
        Sum
    }
}
