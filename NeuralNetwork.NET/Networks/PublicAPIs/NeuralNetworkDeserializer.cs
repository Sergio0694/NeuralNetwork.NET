using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
using Newtonsoft.Json;

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
                dynamic deserialized = JsonConvert.DeserializeObject(json);
                int inputs = deserialized.Inputs, outputs = deserialized.Outputs;
                int[] layersInfo = (int[])deserialized.HiddenLayers.ToObject(typeof(int[]));
                double[][,] weights = (double[][,])deserialized.Weights.ToObject(typeof(double[][,]));

                //Checks
                if (weights.Length == 0) return null; // Missing weights info
                if (weights.Length != layersInfo.Length + 1) return null; // Inconsistent layers info
                if (inputs != weights[0].GetLength(0)) return null; // Invalid inputs >> first layer weights
                if (weights[weights.Length - 1].GetLength(1) != outputs) return null; // Invalid last layer >> output weights
                for (int i = 0; i < weights.Length - 1; i++)
                    if (weights[i].GetLength(1) != weights[i + 1].GetLength(0)) return null; // Inconsistent weights

                // Parse the right network type
                if (weights.Length == 1)
                    return new LinearPerceptron(inputs, outputs, weights[0]);
                if (weights.Length == 2)
                    return new NeuralNetwork(weights[0], weights[1]);
                return null;
            }
            catch
            {
                // Invalid JSON
                return null;
            }
        }
    }
}
