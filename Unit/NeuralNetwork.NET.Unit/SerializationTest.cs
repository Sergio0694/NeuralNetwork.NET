using System;
using System.IO;
using System.Reflection;
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
            INeuralNetwork network = new NeuralNetwork(
                NetworkLayers.Convolutional(new VolumeInformation(28, 1), 5, 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(new VolumeInformation(24, 20), ActivationFunctionType.ReLU),
                NetworkLayers.Convolutional(new VolumeInformation(12, 20), 5, 10, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(new VolumeInformation(8, 10), ActivationFunctionType.ReLU),
                NetworkLayers.FullyConnected(160, 8, ActivationFunctionType.Sigmoid), 
                NetworkLayers.FullyConnected(8, 4, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            String json = network.SerializeAsJSON();
            INeuralNetwork copy = NeuralNetworkLoader.TryLoad(json);
            Assert.IsTrue(copy != null);
            Assert.IsTrue(copy.Equals(network));
        }

        [TestMethod]
        public void BinarySerialize1()
        {
            INeuralNetwork network = new NeuralNetwork(
                NetworkLayers.FullyConnected(784, 30, ActivationFunctionType.Sigmoid),
                NetworkLayers.FullyConnected(30, 10, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            String folderPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            network.Save(new DirectoryInfo(folderPath), "test1");
            INeuralNetwork copy = NeuralNetworkLoader.TryLoad($"{Path.Combine(folderPath, "test1")}.nnet");
            Assert.IsTrue(copy != null);
            Assert.IsTrue(copy.Equals(network));
        }

        [TestMethod]
        public void BinarySerialize2()
        {
            INeuralNetwork network = new NeuralNetwork(
                NetworkLayers.Convolutional(new VolumeInformation(28, 1), 5, 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(new VolumeInformation(24, 20), ActivationFunctionType.ReLU),
                NetworkLayers.Convolutional(new VolumeInformation(12, 20), 5, 10, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(new VolumeInformation(8, 10), ActivationFunctionType.ReLU),
                NetworkLayers.FullyConnected(160, 8, ActivationFunctionType.Sigmoid),
                NetworkLayers.FullyConnected(8, 4, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            String folderPath = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            network.Save(new DirectoryInfo(folderPath), "test");
            INeuralNetwork copy = NeuralNetworkLoader.TryLoad($"{Path.Combine(folderPath, "test")}.nnet");
            Assert.IsTrue(copy != null);
            Assert.IsTrue(copy.Equals(network));
        }
    }
}
