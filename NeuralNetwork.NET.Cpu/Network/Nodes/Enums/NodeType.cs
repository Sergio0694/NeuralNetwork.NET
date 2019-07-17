namespace NeuralNetworkDotNet.Network.Nodes.Enums
{
    /// <summary>
    /// An <see langword="enum"/> that indicates the type of a particular node
    /// </summary>
    public enum NodeType
    {
        /// <summary>
        /// Maps to <see cref="Nullary.PlaceholderNode"/>
        /// </summary>
        Placeholder,

        /// <summary>
        /// Maps to <see cref="Unary.ActivationNode"/>
        /// </summary>
        Activation,

        /// <summary>
        /// Maps to <see cref="Unary.BatchNormalizationNode"/>
        /// </summary>
        BatchNormalization,

        /// <summary>
        /// Maps to <see cref="Unary.ConvolutionalNode"/>
        /// </summary>
        Convolution,

        /// <summary>
        /// Maps to <see cref="Unary.DropoutNode"/>
        /// </summary>
        Dropout,

        /// <summary>
        /// Maps to <see cref="Unary.FullyConnectedNode"/>
        /// </summary>
        FullyConnected,

        /// <summary>
        /// Maps to <see cref="Unary.PoolingNode"/>
        /// </summary>
        Pooling,

        /// <summary>
        /// Maps to <see cref="Unary.Losses.OutputNode"/>
        /// </summary>
        Output,

        /// <summary>
        /// Maps to <see cref="Unary.Losses.SoftmaxNode"/>
        /// </summary>
        Softmax,

        /// <summary>
        /// Maps to <see cref="Binary.SumNode"/>
        /// </summary>
        Sum,

        /// <summary>
        /// Maps to <see cref="Binary.DepthConcatenationNode"/>
        /// </summary>
        DepthConcatenation,
    }
}
