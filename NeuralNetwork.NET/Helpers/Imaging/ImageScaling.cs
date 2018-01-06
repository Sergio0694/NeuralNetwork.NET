namespace NeuralNetworkNET.Helpers.Imaging
{
    /// <summary>
    /// Indicates the quality used to export the weights of a neural network as an image
    /// </summary>
    public enum ImageScaling
    {
        /// <summary>
        /// The weights are exported using a 1:1 pixel size ratio
        /// </summary>
        Native,

        /// <summary>
        /// The weights are upscaled in size when exported as images
        /// </summary>
        HighQuality
    }
}