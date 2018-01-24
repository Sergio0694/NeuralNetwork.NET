namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates the type of a given neural network
    /// </summary>
    public enum NetworkType : byte
    {
        /// <summary>
        /// The classic network model, consisting in a linear stack of connected layers
        /// </summary>
        Sequential,

        /// <summary>
        /// A network with a custom computation graph used to process its inputs
        /// </summary>
        ComputationGraph
    }
}
