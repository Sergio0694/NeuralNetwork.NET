namespace NeuralNetworkLibrary.Networks.PublicAPIs
{
    /// <summary>
    /// An interface to mask a neural network implementation
    /// </summary>
    public interface INeuralNetwork
    {
        /// <summary>
        /// Gets the size of the input layer
        /// </summary>
        int InputLayerSize { get; }

        /// <summary>
        /// Gets the size of the output layer
        /// </summary>
        int OutputLayerSize { get; }

        /// <summary>
        /// Forwards the input through the network
        /// </summary>
        /// <param name="input">The input to process</param>
        double[,] Forward(double[,] input);

        /// <summary>
        /// Serializes the current instance into a byte array
        /// </summary>
        byte[] Serialize();
    }
}