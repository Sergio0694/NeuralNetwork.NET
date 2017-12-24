namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the type of a neural network layer (for serialization purposes only)
    /// </summary>
    public enum LayerType : byte
    {
        /// <summary>
        /// A fully connected layer, mapping n inputs to m outputs
        /// </summary>
        FullyConnected,

        /// <summary>
        /// A convolutional layer, which keeps spatial information on the input volume
        /// </summary>
        Convolutional,

        /// <summary>
        /// A pooling layer, useful to reduce the size of the input data volume
        /// </summary>
        Pooling,

        /// <summary>
        /// A fully connected output layer, with an arbitrary activation and cost function
        /// </summary>
        Output,

        /// <summary>
        /// A softmax layer, with the softmax activation and log-likelyhood cost function
        /// </summary>
        Softmax,

        /// <summary>
        /// An inception module, combining different kinds of convolution with a pooling operation
        /// </summary>
        Inception
    }
}