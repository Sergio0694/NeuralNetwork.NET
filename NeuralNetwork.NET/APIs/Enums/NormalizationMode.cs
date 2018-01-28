namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see langword="enum"/> indicating the normalization mode to apply to the input data of a layer
    /// </summary>
    public enum NormalizationMode : byte
    {
        /// <summary>
        /// Spatial normalization, with a single mean and variance value per input channel (feature map)
        /// </summary>
        Spatial,

        /// <summary>
        /// Activation-wise normalization, with a separate mean and variance value per activation
        /// </summary>
        PerActivation
    }
}