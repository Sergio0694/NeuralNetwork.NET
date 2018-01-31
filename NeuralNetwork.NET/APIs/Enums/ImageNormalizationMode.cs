namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the type of normalization to apply to a loaded image
    /// </summary>
    public enum ImageNormalizationMode : byte
    {
        /// <summary>
        /// The individual pixel values are mapped in the [0,1] range
        /// </summary>
        Sigmoid,

        /// <summary>
        /// The individual pixel values are mapped in the [-1,1] range
        /// </summary>
        Normal,

        /// <summary>
        /// No normalization is applied, and all the pixel values are loaded with their original value
        /// </summary>
        None
    }
}