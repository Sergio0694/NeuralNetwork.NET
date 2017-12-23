using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Extensions;
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
            using (FileStream stream = file.OpenRead())
                return TryLoad(stream);
        }

        /// <summary>
        /// Tries to deserialize a network from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> instance for the network to load</param>
        /// <param name="deserializers">The list of deserializers to use to load the input network</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] Stream stream, params LayerDeserializer[] deserializers)
        {
            if (deserializers.GroupBy(f => f).Any(g => g.Count() > 1)) throw new ArgumentException("The deserializers list can't contain duplicate entries", nameof(deserializers));
            try
            {
                List<INetworkLayer> layers = new List<INetworkLayer>();
                using (GZipStream gzip = new GZipStream(stream, CompressionMode.Decompress))
                {
                    while (gzip.TryRead(out LayerType type))
                    {
                        // Process the deserializers in precedence order
                        INetworkLayer layer = null;
                        foreach (LayerDeserializer deserializer in deserializers)
                        {
                            layer = deserializer(gzip, type);
                            if (layer != null) break;
                        }

                        // Process the layer
                        layers.Add(layer ?? DefaultLayersDeserializer(gzip, type));
                    }
                }

                // Try to create the network to return
                return new NeuralNetwork(layers.ToArray());
            }
            catch
            {
                // Locked or invalid file
                return null;
            }
        }

        // Default layers deserializer
        [MustUseReturnValue, NotNull]
        private static INetworkLayer DefaultLayersDeserializer([NotNull] Stream stream, LayerType type)
        {
            switch (type)
            {
                case LayerType.FullyConnected: return FullyConnectedLayer.Deserialize(stream);
                case LayerType.Convolutional: return ConvolutionalLayer.Deserialize(stream);
                case LayerType.Pooling: return PoolingLayer.Deserialize(stream);
                case LayerType.Output: return OutputLayer.Deserialize(stream);
                case LayerType.Softmax: return SoftmaxLayer.Deserialize(stream);
                default: throw new ArgumentOutOfRangeException(nameof(type), $"The {type} layer type is not supported by the default deserializer");
            }
        }
    }
}
