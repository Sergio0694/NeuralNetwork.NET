using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.Networks.Layers.Cuda;

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
        /// <param name="preference">The layers deserialization preference</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] FileInfo file, LayersLoadingPreference preference)
        {
            using (FileStream stream = file.OpenRead())
                return TryLoad(stream, preference);
        }

        /// <summary>
        /// Tries to deserialize a network from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> instance for the network to load</param>
        /// <param name="preference">The layers deserialization preference</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] Stream stream, LayersLoadingPreference preference)
        {
            try
            {
                List<INetworkLayer> layers = new List<INetworkLayer>();
                using (GZipStream gzip = new GZipStream(stream, CompressionMode.Decompress))
                {
                    while (gzip.TryRead(out LayerType type))
                    {
                        // Deserialization attempt
                        INetworkLayer layer = null;
                        if (preference == LayersLoadingPreference.Cuda) layer = CudaDeserialize(gzip, type);
                        if (layer == null) layer = CpuDeserialize(gzip, type);
                        if (layer == null) return null;

                        // Add to the queue
                        layers.Add(layer);
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

        #region Deserializers

        // Default layers deserializer
        [MustUseReturnValue, CanBeNull]
        private static INetworkLayer CpuDeserialize([NotNull] Stream stream, LayerType type)
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

        // Cuda layers deserializer
        [MustUseReturnValue, CanBeNull]
        private static INetworkLayer CudaDeserialize([NotNull] Stream stream, LayerType type)
        {
            switch (type)
            {
                case LayerType.FullyConnected: return CuDnnFullyConnectedLayer.Deserialize(stream);
                case LayerType.Convolutional: return CuDnnConvolutionalLayer.Deserialize(stream);
                case LayerType.Pooling: return CuDnnPoolingLayer.Deserialize(stream);
                case LayerType.Softmax: return CuDnnSoftmaxLayer.Deserialize(stream);
                case LayerType.Inception: return CuDnnInceptionLayer.Deserialize(stream);
                default: return null;
            }
        }
        
        #endregion
    }
}
