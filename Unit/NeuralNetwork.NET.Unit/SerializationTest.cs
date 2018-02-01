using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.Networks.Layers.Initialization;
using SixLabors.ImageSharp.PixelFormats;

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
            PoolingInfo info = PoolingInfo.New(PoolingMode.AverageIncludingPadding, 3, 3, 1, 1);
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
                float[] w = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(784), 30, WeightsInitializationMode.GlorotNormal);
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
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Rgb24>(120, 120),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.AbsoluteReLU),
                NetworkLayers.Convolutional((5, 5), 20, ActivationType.ELU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.Identity),
                NetworkLayers.Pooling(ActivationType.ReLU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.Identity),
                NetworkLayers.Pooling(ActivationType.ReLU),
                NetworkLayers.FullyConnected(125, ActivationType.Tanh),
                NetworkLayers.Softmax(133));
            using (MemoryStream stream = new MemoryStream())
            {
                network.Save(stream);
                stream.Seek(0, SeekOrigin.Begin);
                INeuralNetwork copy = NetworkLoader.TryLoad(stream, ExecutionModePreference.Cpu);
                Assert.IsTrue(network.Equals(copy));
            }
        }

        [TestMethod]
        public void JsonMetadataSerialization()
        {
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Rgb24>(120, 120),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.AbsoluteReLU),
                NetworkLayers.Convolutional((5, 5), 20, ActivationType.ELU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.Identity),
                NetworkLayers.Pooling(ActivationType.ReLU),
                NetworkLayers.Convolutional((10, 10), 20, ActivationType.Identity),
                NetworkLayers.Pooling(ActivationType.Identity),
                NetworkLayers.BatchNormalization(NormalizationMode.Spatial, ActivationType.ReLU),
                NetworkLayers.FullyConnected(125, ActivationType.Tanh),
                NetworkLayers.Softmax(133));
            String metadata1 = network.SerializeMetadataAsJson();
            Assert.IsTrue(metadata1.Length > 0);
            Assert.IsTrue(metadata1.Equals(network.Clone().SerializeMetadataAsJson()));
            network.Layers.First().To<INetworkLayer, ConvolutionalLayer>().Weights[0] += 0.1f;
            Assert.IsFalse(metadata1.Equals(network.SerializeMetadataAsJson()));
        }
    }
}
