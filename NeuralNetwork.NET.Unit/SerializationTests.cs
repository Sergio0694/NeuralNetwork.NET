using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(SerializationTests))]
    public class SerializationTests
    {
        [TestMethod]
        public void BinarySerialize()
        {
            NeuralNetwork network = NeuralNetwork.NewRandom(5, 8, 4);
            double[] data = network.Serialize();
            NeuralNetwork copy = NeuralNetwork.Deserialize(data, 5, 8, 4);
            Assert.IsTrue(copy.Equals(network));
        }

        [TestMethod]
        public void BinaryDeserialize()
        {
            NeuralNetwork network = NeuralNetwork.NewRandom(5, 8, 4);
            double[] data = network.Serialize();
            Assert.ThrowsException<InvalidOperationException>(() => NeuralNetwork.Deserialize(data, 5, 7, 4));
        }

        [TestMethod]
        public void JsonSerialize()
        {
            NeuralNetwork network = NeuralNetwork.NewRandom(5, 8, 4);
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
