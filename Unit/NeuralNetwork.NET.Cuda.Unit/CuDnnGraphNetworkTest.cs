using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.Networks.Layers.Cuda;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// A class with some test methods for the <see cref="Networks.Implementations.ComputationGraphNetwork"/> class with cuDNN layers
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CuDnnGraphNetworkTest))]
    public class CuDnnGraphNetworkTest
    {
        private static void ForwardTest([NotNull] INeuralNetwork n1, [NotNull] INeuralNetwork n2)
        {
            float[,] x = new float[257, n1.InputInfo.Size];
            for (int i = 0; i < 257; i++)
                for (int j = 0; j < n1.InputInfo.Size; j++)
                    x[i, j] = ThreadSafeRandom.NextFloat();
            float[,]
                y1 = n1.Forward(x),
                y2 = n2.Forward(x);
            Assert.IsTrue(y1.ContentEquals(y2));
        }

        [TestMethod]
        public void ForwardTest1()
        {
            INeuralNetwork cpu = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(28, 28), root =>
            {
                var fc1 = root.Layer(NetworkLayers.FullyConnected(100, ActivationType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            INeuralNetwork gpu = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(28, 28), root =>
            {
                var fc1l = cpu.Layers[0].To<INetworkLayer, FullyConnectedLayer>();
                var fc1 = root.Layer(_ => new CuDnnFullyConnectedLayer(fc1l.InputInfo, 100, fc1l.Weights, fc1l.Biases, fc1l.ActivationType));
                var sm1l = cpu.Layers[1].To<INetworkLayer, SoftmaxLayer>();
                fc1.Layer(_ => new CuDnnSoftmaxLayer(sm1l.InputInfo, sm1l.OutputInfo.Size, sm1l.Weights, sm1l.Biases));
            });
            ForwardTest(cpu, gpu);
        }

        [TestMethod]
        public void ForwardTest2()
        {
            INeuralNetwork cpu = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (5, 5), 20, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.Default, ActivationType.ReLU),
                CuDnnNetworkLayers.Convolutional(ConvolutionInfo.Default, (3, 3), 20, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(PoolingInfo.Default, ActivationType.ReLU),
                CuDnnNetworkLayers.FullyConnected(100, ActivationType.LeCunTanh),
                CuDnnNetworkLayers.FullyConnected(50, ActivationType.LeCunTanh),
                CuDnnNetworkLayers.Softmax(10));
            INeuralNetwork gpu = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(28, 28), root =>
            {
                var conv1 = root.Layer(_ => cpu.Layers[0].Clone());
                var pool1 = conv1.Layer(_ => cpu.Layers[1].Clone());
                var conv2 = pool1.Layer(_ => cpu.Layers[2].Clone());
                var pool2 = conv2.Layer(_ => cpu.Layers[3].Clone());
                var fc1 = pool2.Layer(_ => cpu.Layers[4].Clone());
                var fc2 = fc1.Layer(_ => cpu.Layers[5].Clone());
                fc2.Layer(_ => cpu.Layers[6].Clone());
            });
            ForwardTest(cpu, gpu);
        }
    }
}
