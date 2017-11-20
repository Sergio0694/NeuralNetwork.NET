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
                NetworkLayers.Convolutional(new VolumeInformation(28, 1), 5, 20, ActivationFunctionType.ReLU),
                NetworkLayers.Convolutional(new VolumeInformation(24, 20), 5, 10, ActivationFunctionType.ReLU),
                NetworkLayers.Pooling(new VolumeInformation(20, 10)),
                NetworkLayers.FullyConnected(100, 8, ActivationFunctionType.Sigmoid), 
                NetworkLayers.FullyConnected(8, 4, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            String json = network.SerializeAsJSON();
            INeuralNetwork copy = NeuralNetworkDeserializer.TryDeserialize(json);
            Assert.IsTrue(copy != null);
            Assert.IsTrue(copy.Equals(network));
        }
    }
}
