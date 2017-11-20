using System;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// A static class that handles the JSON deserialization for the neural networks
    /// </summary>
    public static class NeuralNetworkDeserializer
    {
        /// <summary>
        /// Tries to deserialize a network from the input JSON text
        /// </summary>
        /// <param name="json">The source JSON data to parse</param>
        /// <returns>The deserialized network, or null if the operation fails</returns>
        [PublicAPI]
        [CanBeNull]
        public static INeuralNetwork TryDeserialize([NotNull] String json)
        {
            try
            {
                // Get the general parameters and the hidden layers info
                JObject jObject = (JObject)JsonConvert.DeserializeObject(json);
                INetworkLayer[] layers = jObject["Layers"].Select<JToken, INetworkLayer>(layer =>
                {
                    if (!Enum.TryParse(layer["LayerType"].ToString(), out LayerType type))
                        throw new InvalidOperationException("Unsupported JSON network");
                    switch (type)
                    {
                        case LayerType.FullyConnected:
                            return new FullyConnectedLayer(
                                layer["Weights"].ToObject<float[,]>(),
                                layer["Biases"].ToObject<float[]>(),
                                layer["ActivationFunctionType"].ToObject<ActivationFunctionType>());
                        case LayerType.Convolutional:
                            return new ConvolutionalLayer(
                                layer["InputVolume"].ToObject<VolumeInformation>(),
                                layer["KernelVolume"].ToObject<VolumeInformation>(),
                                layer["OutputVolume"].ToObject<VolumeInformation>(),
                                layer["Weights"].ToObject<float[,]>(),
                                layer["Biases"].ToObject<float[]>(),
                                layer["ActivationFunctionType"].ToObject<ActivationFunctionType>());
                        case LayerType.Pooling:
                            return new PoolingLayer(layer["InputVolume"].ToObject<VolumeInformation>());
                        case LayerType.Output:
                            return new OutputLayer(
                                layer["Weights"].ToObject<float[,]>(),
                                layer["Biases"].ToObject<float[]>(),
                                layer["ActivationFunctionType"].ToObject<ActivationFunctionType>(),
                                layer["CostFunctionType"].ToObject<CostFunctionType>());
                        case LayerType.Softmax:
                            return new SoftmaxLayer(
                                layer["Weights"].ToObject<float[,]>(),
                                layer["Biases"].ToObject<float[]>());
                        default:
                            throw new InvalidOperationException("Unsupported JSON network");
                    }
                }).ToArray();
                INeuralNetwork network = new NeuralNetwork(layers);
                return json.Equals(network.SerializeAsJSON()) ? network : null;
            }
            catch
            {
                // Invalid JSON
                return null;
            }
        }
    }
}
