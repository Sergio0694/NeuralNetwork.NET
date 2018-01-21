namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see langword="enum"/> that indicates the type of a node in a graph network
    /// </summary>
    public enum ComputationGraphNodeType : byte
    {
        /// <summary>
        /// The root node for a computation graph, that forwards the network inputs through the computation pipeline(s)
        /// </summary>
        Input,

        /// <summary>
        /// A computation graph node with an associated <see cref="Interfaces.INetworkLayer"/> that processes an input <see cref="Structs.Tensor"/>
        /// </summary>
        Processing,

        /// <summary>
        /// The root node for a training sub-graph, a secondary training branch used during backpropagation to inject partial gradients
        /// </summary>
        TrainingBranch,

        /// <summary>
        /// A computation graph node that merges a series of inputs by stacking them along the depth axis
        /// </summary>
        DepthConcatenation,

        /// <summary>
        /// A computation graph node that merges a series of inputs by summing their values together
        /// </summary>
        Sum
    }
}
