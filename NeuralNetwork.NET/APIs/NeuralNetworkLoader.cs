using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Implementations.Layers;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that handles the JSON deserialization for the neural networks
    /// </summary>
    public static class NeuralNetworkLoader
    {
        /// <summary>
        /// Gets the file extension used when saving a network
        /// </summary>
        public const String NetworkFileExtension = ".nnet";

        /// <summary>
        /// Tries to deserialize a network from the input file
        /// </summary>
        /// <param name="file">The <see cref="FileInfo"/> instance for the file to load</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] FileInfo file)
        {
            try
            {
                using (Stream stream = file.OpenRead())
                {
                    throw new NotImplementedException();
                }
            }
            catch
            {
                // Locked or invalid file
                return null;
            }
        }
    }
}
