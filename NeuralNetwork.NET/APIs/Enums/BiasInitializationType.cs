namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see langword="enum"/> indicating an initialization mode for the biases in a network layer
    /// </summary>
    public enum BiasInitializationMode : byte
    {
        /// <summary>
        /// All the bias values are initially set to 0
        /// </summary>
        Zero,

        /// <summary>
        /// Every bias value is assigned from a gaussian distribution ~N(0, 1)
        /// </summary>
        Gaussian
    }
}
