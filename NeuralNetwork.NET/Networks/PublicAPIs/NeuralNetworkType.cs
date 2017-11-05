namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// Indicates the type of a neural network
    /// </summary>
    public enum NeuralNetworkType
    {
        /// <summary>
        /// The neural network doesn't have additional biases
        /// </summary>
        Unbiased,

        /// <summary>
        /// The neural network has a bias value for each neuron, starting from the second layer
        /// </summary>
        Biased
    }
}