using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Activations;
using System.IO;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN layers serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(SerializationTest))]
    public class SerializationTest
    {
        [TestMethod]
        public void NetworkSerialization()
        {
            INeuralNetwork network = NetworkManager.NewNetwork(TensorInfo.CreateForRgbImage(120, 120),
                t => CuDnnNetworkLayers.Convolutional(t, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (10, 10), 20, ActivationFunctionType.AbsoluteReLU),
                t => CuDnnNetworkLayers.Convolutional(t, ConvolutionInfo.New(ConvolutionMode.Convolution, 2, 2), (5, 5), 20, ActivationFunctionType.ELU),
                t => CuDnnNetworkLayers.Convolutional(t, ConvolutionInfo.Default, (10, 10), 20, ActivationFunctionType.Identity),
                t => CuDnnNetworkLayers.Pooling(t, PoolingInfo.New(PoolingMode.AverageIncludingPadding, 2, 2, 1, 1), ActivationFunctionType.ReLU),
                t => CuDnnNetworkLayers.Convolutional(t, ConvolutionInfo.Default, (10, 10), 20, ActivationFunctionType.Identity),
                t => CuDnnNetworkLayers.Pooling(t, PoolingInfo.Default, ActivationFunctionType.ReLU),
                t => CuDnnNetworkLayers.FullyConnected(t, 125, ActivationFunctionType.Tanh),
                t => CuDnnNetworkLayers.FullyConnected(t, 27, ActivationFunctionType.Tanh),
                t => CuDnnNetworkLayers.Softmax(t, 133));
            using (MemoryStream stream = new MemoryStream())
            {
                network.Save(stream);
                stream.Seek(0, SeekOrigin.Begin);
                INeuralNetwork copy = NeuralNetworkLoader.TryLoad(stream, CuDnnNetworkLayersDeserializer.Deserializer);
                Assert.IsTrue(network.Equals(copy));
            }
        }
    }
}
