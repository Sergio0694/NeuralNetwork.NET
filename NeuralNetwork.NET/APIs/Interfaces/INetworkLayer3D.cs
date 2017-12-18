using NeuralNetworkNET.APIs.Misc;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An interface for an <see cref="INetworkLayer"/> instance that processes a data volume
    /// </summary>
    public interface INetworkLayer3D
    {
        /// <summary>
        /// Gets the input data volume for the network
        /// </summary>
        TensorInfo InputInfo { get; }

        /// <summary>
        /// Gets the output data volume for the network
        /// </summary>
        TensorInfo OutputInfo { get; }
    }
}