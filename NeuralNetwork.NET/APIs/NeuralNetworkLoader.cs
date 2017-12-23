using System;
using System.IO;
using System.IO.Compression;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Extensions;

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
            using (FileStream stream = file.OpenRead())
                return TryLoad(stream);
        }

        /// <summary>
        /// Tries to deserialize a network from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> instance for the network to load</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] Stream stream)
        {
            try
            {
                using (GZipStream gzip = new GZipStream(stream, CompressionMode.Decompress))
                {
                    while (gzip.TryRead(out LayerType type))
                    {

                    }
                }
                throw new NotImplementedException();
            }
            catch
            {
                // Locked or invalid file
                return null;
            }
        }
    }
}
