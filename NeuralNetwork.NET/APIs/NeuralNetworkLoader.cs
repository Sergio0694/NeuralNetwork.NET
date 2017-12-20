using System;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
        /// Tries to deserialize a network from the input JSON text
        /// </summary>
        /// <param name="json">The source JSON data to parse</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoadJson([NotNull] String json)
        {
            throw new NotImplementedException("Json serialization is not supported");
        }

        /// <summary>
        /// Tries to deserialize a network from the input file
        /// </summary>
        /// <param name="path">The path to the file to load</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [Pure, CanBeNull]
        public static INeuralNetwork TryLoad(String path)
        {
            // Json
            if (!Path.GetExtension(path).Equals(NetworkFileExtension))
                return Path.GetExtension(path).Equals(".json")
                    ? TryLoadJson(File.ReadAllText(path))
                    : null;

            // Binary
            try
            {
                using (Stream stream = File.OpenRead(path))
                    return TryLoad(stream);
            }
            catch
            {
                // Locked or invalid file
                return null;
            }
        }

        // Private binary loader
        private static INeuralNetwork TryLoad(Stream stream)
        {
            INetworkLayer[] layers = new INetworkLayer[stream.ReadInt32()];
            for (int i = 0; i < layers.Length; i++)
            {
                LayerType type = (LayerType)stream.ReadByte();
                ActivationFunctionType activation = (ActivationFunctionType)stream.ReadByte();
                int
                    inputs = stream.ReadInt32(),
                    outputs = stream.ReadInt32();
                switch (type)
                {
                    case LayerType.FullyConnected:
                        layers[i] = new FullyConnectedLayer(stream.ReadFloatArray(inputs, outputs), stream.ReadFloatArray(outputs), activation);
                        break;
                    case LayerType.Convolutional:
                        throw new NotImplementedException("convolution deserialization not implemented yet");
                        /*
                        TensorInfo
                            inVolume = new TensorInfo(stream.ReadInt32(), stream.ReadInt32(), stream.ReadInt32()),
                            outVolume = new TensorInfo(stream.ReadInt32(), stream.ReadInt32(), stream.ReadInt32()),
                            kVolume = new TensorInfo(stream.ReadInt32(), stream.ReadInt32(), stream.ReadInt32());
                        layers[i] = new ConvolutionalLayer(inVolume, kVolume, outVolume,
                            stream.ReadFloatArray(outVolume.Channels, kVolume.Size),
                            stream.ReadFloatArray(outVolume.Channels), activation); */
                    case LayerType.Pooling:
                        throw new NotImplementedException("Pooling deserialization not implemented yet");
                        //layers[i] = new PoolingLayer(new TensorInfo(stream.ReadInt32(), stream.ReadInt32(), stream.ReadInt32()), activation);
                    case LayerType.Output:
                        layers[i] = new OutputLayer(stream.ReadFloatArray(inputs, outputs), stream.ReadFloatArray(outputs), activation, (CostFunctionType)stream.ReadByte());
                        break;
                    case LayerType.Softmax:
                        layers[i] = new SoftmaxLayer(stream.ReadFloatArray(inputs, outputs), stream.ReadFloatArray(outputs));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return new NeuralNetwork(layers);
        }
    }
}
