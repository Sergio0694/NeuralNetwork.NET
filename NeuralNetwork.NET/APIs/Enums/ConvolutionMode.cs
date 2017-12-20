namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// A simple <see cref="enum"/> indicating the type of convolution operation to perform
    /// </summary>
    public enum ConvolutionMode
    {
        /// <summary>
        /// The default convolution mode, with the kernel taargeting pixels in the opposite position
        /// </summary>
        Convolution = 0,

        /// <summary>
        /// The cross-correlation mode (equivalent to a convolution with a flipped kernel)
        /// </summary>
        CrossCorrelation = 1
    }
}
