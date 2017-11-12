namespace NeuralNetworkNET.Convolution.Operations
{
    /// <summary>
    /// Indicates a convolution operation to perform
    /// </summary>
    public enum ConvolutionOperationType : byte
    {
        /// <summary>
        /// A convolution with a 3x3 kernel
        /// </summary>
        Convolution3x3,

        /// <summary>
        /// The ReLU function application
        /// </summary>
        ReLU,

        /// <summary>
        /// A pooling operation with a 2x2 window
        /// </summary>
        Pool2x2,

        /// <summary>
        /// A normalization function that maps all the values in the [0..1] range
        /// </summary>
        Normalization
    }
}