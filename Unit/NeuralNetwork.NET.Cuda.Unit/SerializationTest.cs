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
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.CreateForRgbImage(120, 120),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (10, 10), 20, ActivationFunctionType.AbsoluteReLU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.New(ConvolutionMode.Convolution, 2, 2), (5, 5), 20, ActivationFunctionType.ELU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (10, 10), 20, ActivationFunctionType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.New(PoolingMode.AverageIncludingPadding, 2, 2, 1, 1), ActivationFunctionType.ReLU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (10, 10), 20, ActivationFunctionType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.Default, ActivationFunctionType.ReLU),
                CuDnnNetworkLayers.FullyConnected(125, ActivationFunctionType.Tanh),
                CuDnnNetworkLayers.FullyConnected(27, ActivationFunctionType.Tanh),
                CuDnnNetworkLayers.Softmax(133));
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
