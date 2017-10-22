using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
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
                int inputs = jObject[nameof(INeuralNetwork.InputLayerSize)].ToObject<int>(), outputs = jObject[nameof(INeuralNetwork.OutputLayerSize)].ToObject<int>();
                int[] layersInfo = jObject[nameof(INeuralNetwork.HiddenLayers)].ToObject<int[]>();
                double[][,] weights = jObject["Weights"].ToObject<double[][,]>();
                double[][] biases = jObject["Biases"].ToObject<double[][]>();

                // Input checks
                if (weights.Length < 1 ||
                    biases.Length < 1 ||
                    inputs != weights[0].GetLength(0) ||
                    outputs != biases[biases.Length - 1].Length) return null;
                for (int i = 0; i < layersInfo.Length; i++)
                    if (layersInfo[i] != biases[i].Length) return null;

                // Try to reconstruct the network
                return new NeuralNetwork(weights, biases);
            }
            catch
            {
                // Invalid JSON
                return null;
            }
        }
    }
}
