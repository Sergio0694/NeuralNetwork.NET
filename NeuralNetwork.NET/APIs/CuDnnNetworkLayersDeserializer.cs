using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Cuda.Layers;
using System.IO;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes a single deserialization method that can be used to load a saved network using the cuDNN layers
    /// </summary>
    public static class CuDnnNetworkLayersDeserializer
    {
        /// <summary>
        /// Gets the <see cref="LayerDeserializer"/> instance to load cuDNN network layers
        /// </summary>
        [PublicAPI]
        public static LayerDeserializer Deserializer { get; } = Deserialize;

        /// <summary>
        /// Deserializes a layer of the given type from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The <see cref="Stream"/> to use to load the layer data</param>
        /// <param name="type">The type of network layer to return</param>
        private static INetworkLayer Deserialize([NotNull] Stream stream, LayerType type)
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
    }
}
