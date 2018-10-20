using System;
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
    [PublicAPI]
    public static class NetworkLoader
    {
        /// <summary>
        /// Gets the file extension used when saving a network
        /// </summary>
        public const string NetworkFileExtension = ".nnet";

        /// <summary>
        /// Tries to deserialize a network from the input file
        /// </summary>
        /// <param name="file">The <see cref="FileInfo"/> instance for the file to load</param>
        /// <param name="preference">The layers deserialization preference</param>
        /// <returns>The deserialized network, or <see langword="null"/> if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] FileInfo file, ExecutionModePreference preference)
        {
            try
            {
                using (FileStream stream = file.OpenRead())
                    return TryLoad(stream, preference);
            }
            catch (FileNotFoundException)
            {
                // Just keep going
                return null;
            }
        }

        /// <summary>
        /// Tries to deserialize a network from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> instance for the network to load</param>
        /// <param name="preference">The layers deserialization preference</param>
        /// <returns>The deserialized network, or <see langword="null"/> if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad([NotNull] Stream stream, ExecutionModePreference preference)
        {
            try
            {
                using (GZipStream gzip = new GZipStream(stream, CompressionMode.Decompress))
                {
                    if (!gzip.TryRead(out NetworkType model)) return null;
                    switch (model)
                    {
                        case NetworkType.Sequential: return SequentialNetwork.Deserialize(gzip, preference);
                        case NetworkType.ComputationGraph: return ComputationGraphNetwork.Deserialize(gzip, preference);
                        default: return null;
                    }
                }
            }
            catch
            {
                // Locked or invalid file
                return null;
            }
        }

        #region Deserializers

        /// <summary>
        /// Tries to deserialize a CPU-powered network layer
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="type">The target network layer type</param>
        [MustUseReturnValue, CanBeNull]
        internal static INetworkLayer CpuLayerDeserialize([NotNull] Stream stream, LayerType type)
        {
            switch (type)
            {
                case LayerType.FullyConnected: return FullyConnectedLayer.Deserialize(stream);
                case LayerType.Convolutional: return ConvolutionalLayer.Deserialize(stream);
                case LayerType.Pooling: return PoolingLayer.Deserialize(stream);
                case LayerType.Output: return OutputLayer.Deserialize(stream);
                case LayerType.Softmax: return SoftmaxLayer.Deserialize(stream);
                case LayerType.BatchNormalization: return BatchNormalizationLayer.Deserialize(stream);
                default: throw new ArgumentOutOfRangeException(nameof(type), $"The {type} layer type is not supported by the default deserializer");
            }
        }

        /// <summary>
        /// Tries to deserialize a Cuda-powered network layer
        /// </summary>
        /// <param name="stream">The source <see cref="Stream"/></param>
        /// <param name="type">The target network layer type</param>
        [MustUseReturnValue, CanBeNull]
        internal static INetworkLayer CuDnnLayerDeserialize([NotNull] Stream stream, LayerType type)
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
