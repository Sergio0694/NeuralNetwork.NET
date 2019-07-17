using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;

namespace NeuralNetworkDotNet.APIs.Interfaces
{
    /// <summary>
    /// An <see langword="interface"/> for a neural network
    /// </summary>
    public interface INetwork : IEquatable<INetwork>, IClonable<INetwork>
    {
        /// <summary>
        /// Gets the <see cref="Shape"/> of the network inputs
        /// </summary>
        Shape InputShape { get; }

        /// <summary>
        /// Gets the <see cref="Shape"/> of the network outputs
        /// </summary>
        Shape OutputShape { get; }

        /// <summary>
        /// Gets the number of nodes in the current network
        /// </summary>
        int NodesCount { get; }

        /// <summary>
        /// Gets the number of parameters in the current network
        /// </summary>
        int ParametersCount { get; }

        /// <summary>
        /// Gets whether or not there is a numeric overflow in the network
        /// </summary>
        bool IsInNumericOverflow { get; }

        /// <summary>
        /// Forwards an input <see cref="Tensor"/> through the network
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        /// <returns>The processed output <see cref="Tensor"/> for the current network</returns>
        [Pure, NotNull]
        Tensor Forward([NotNull] Tensor x);

        /// <summary>
        /// Calculates the loss of the network for a pair of input tensors
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        /// <param name="y">The expected output <see cref="Tensor"/></param>
        /// <returns>The loss of the input according to the given output values</returns>
        [Pure]
        float Loss([NotNull] Tensor x, [NotNull] Tensor y);

        /// <summary>
        /// Serializes the network metadata as a JSON string
        /// </summary>
        [Pure, NotNull]
        string SerializeMetadataAsJson();

        /// <summary>
        /// Saves the network to the target file
        /// </summary>
        /// <param name="path">The path of the target file</param>
        void Save([NotNull] string path);

        /// <summary>
        /// Saves the network to the target stream
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> to use to write the network data</param>
        void Save([NotNull] Stream stream);
    }
}
