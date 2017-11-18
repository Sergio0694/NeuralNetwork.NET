using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(SerializationTest))]
    public class SerializationTest
    {
        [TestMethod]
        public void JsonSerialize()
        {
            NeuralNetwork network = new NeuralNetwork(
                NetworkLayers.FullyConnected(5, 8, ActivationFunctionType.Sigmoid), 
                NetworkLayers.FullyConnected(8, 4, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            String json = network.SerializeAsJSON();
            INeuralNetwork copy = NeuralNetworkDeserializer.TryDeserialize(json);
            Assert.IsTrue(copy != null);
            Assert.IsTrue(copy.Equals(network));
            String faulted = json.Replace("8", "7");
            copy = NeuralNetworkDeserializer.TryDeserialize(faulted);
            Assert.IsTrue(copy == null);
        }
    }
}
