namespace NeuralNetworkNET.Networks.Implementations.Layers.APIs
{
    /// <summary>
    /// An interface for an <see cref="INetworkLayer"/> instance that processes a data volume
    /// </summary>
    public interface INetworkLayer3D
    {
        /// <summary>
        /// Gets the input data volume for the network
        /// </summary>
        VolumeInformation InputVolume { get; }

        /// <summary>
        /// Gets the output data volume for the network
        /// </summary>
        VolumeInformation OutputVolume { get; }
    }
}