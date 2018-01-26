using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using System.IO;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN layers serialization methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CuDnnSerializationTest))]
    public class CuDnnSerializationTest
    {
        [TestMethod]
        public void NetworkSerialization()
        {
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Rgb24>(120, 120),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (10, 10), 20, ActivationType.AbsoluteReLU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.New(ConvolutionMode.Convolution, 2, 2), (5, 5), 20, ActivationType.ELU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (10, 10), 20, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.New(PoolingMode.AverageIncludingPadding, 2, 2, 1, 1), ActivationType.ReLU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (10, 10), 20, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.Default, ActivationType.ReLU),
                CuDnnNetworkLayers.FullyConnected(125, ActivationType.Tanh),
                CuDnnNetworkLayers.FullyConnected(27, ActivationType.Tanh),
                CuDnnNetworkLayers.Softmax(133));
            using (MemoryStream stream = new MemoryStream())
            {
                network.Save(stream);
                stream.Seek(0, SeekOrigin.Begin);
                INeuralNetwork copy = NetworkLoader.TryLoad(stream, ExecutionModePreference.Cuda);
                Assert.IsTrue(network.Equals(copy));
            }
        }
    }
}
