using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers;
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
            NeuralNetwork network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(5), 
                NetworkLayer.FullyConnected(8, ActivationFunctionType.Sigmoid), 
                NetworkLayer.Outputs(4, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
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
