namespace NeuralNetworkNET.Convolution.Misc
{
    /// <summary>
    /// Indicates the convolution mode to use when performing the convolution operation
    /// </summary>
    public enum ConvolutionMode
    {
        /// <summary>
        /// Valid convolution, where the kernel only slides within the margins of the target matrix
        /// </summary>
        Valid,

        /// <summary>
        /// Full convolution, where the kernel slides from each edge of the target matrix, even if it partially exceeds the bounds of that matrix
        /// </summary>
        Full
    }
}