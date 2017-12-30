using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;

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
        public void StructSerialize()
        {
            PoolingInfo info = PoolingInfo.New(PoolingMode.AverageIncludingPadding, 3, 3, 1, 1, 2, 2);
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(info);
                stream.Seek(0, SeekOrigin.Begin);
                Assert.IsTrue(stream.TryRead(out PoolingInfo copy));
                Assert.IsTrue(info.Equals(copy));
            }
        }

        [TestMethod]
        public void EnumSerialize()
        {
            PoolingMode mode = PoolingMode.AverageIncludingPadding;
            using (MemoryStream stream = new MemoryStream())
            {
                stream.Write(mode);
                stream.Seek(0, SeekOrigin.Begin);
                Assert.IsTrue(stream.TryRead(out PoolingMode copy));
                Assert.IsTrue(mode == copy);
            }
        }

        [TestMethod]
        public void StreamSerialize()
        {
            using (MemoryStream stream = new MemoryStream())
            {
                float[] w = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(784), 30, WeightsInitializationMode.GlorotNormal);
                stream.WriteShuffled(w);
                Assert.IsTrue(stream.Position == sizeof(float) * w.Length);
                stream.Seek(0, SeekOrigin.Begin);
                float[] t = stream.ReadUnshuffled(w.Length);
                Assert.IsTrue(w.ContentEquals(t));
            }
        }

        [TestMethod]
        public void NetworkSerialization()
        {
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.CreateForRgbImage(120, 120),
                NetworkLayers.Convolutional((10, 10), 20, ActivationFunctionType.AbsoluteReLU),
                NetworkLayers.Convolutional((5, 5), 20, ActivationFunctionType.ELU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(ActivationFunctionType.ReLU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(ActivationFunctionType.ReLU),
                NetworkLayers.FullyConnected(125, ActivationFunctionType.Tanh),
                NetworkLayers.Softmax(133));
            using (MemoryStream stream = new MemoryStream())
            {
                network.Save(stream);
                stream.Seek(0, SeekOrigin.Begin);
                INeuralNetwork copy = NeuralNetworkLoader.TryLoad(stream);
                Assert.IsTrue(network.Equals(copy));
            }
        }
    }
}
