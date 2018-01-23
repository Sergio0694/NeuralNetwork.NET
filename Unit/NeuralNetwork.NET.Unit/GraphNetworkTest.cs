using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A class with some test methods for the <see cref="ComputationGraphNetwork"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(GraphNetworkTest))]
    public class GraphNetworkTest
    {
        [TestMethod]
        public void GraphNetworkInitialization1()
        {
            INeuralNetwork network = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3reduce1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3 = _3x3reduce1x1.Layer(NetworkLayers.Convolutional((3, 3), 20, ActivationFunctionType.ReLU));

                var stack = _1x1.DepthConcatenation(_3x3);
                var fc1 = stack.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            Assert.IsTrue(network != null);
        }

        [TestMethod]
        public void GraphNetworkInitialization2()
        {
            INeuralNetwork network = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));

                var split = conv1.TrainingBranch();
                split.Layer(NetworkLayers.Softmax(10));

                var fc1 = conv1.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            Assert.IsTrue(network != null);
        }

        [TestMethod]
        public void GraphNetworkInitializationFail1()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3reduce1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3 = _3x3reduce1x1.Layer(NetworkLayers.Convolutional((3, 3), 20, ActivationFunctionType.ReLU));

                var stack = _1x1.DepthConcatenation(_3x3);
                stack.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }
    }
}
