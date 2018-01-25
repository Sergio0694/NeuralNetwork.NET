using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;
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
        #region Initialization

        [TestMethod]
        public void Initialization1()
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
        public void Initialization2()
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

        #endregion

        #region Initializaation errors

        [TestMethod]
        public void InitializationFail1()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3reduce1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3 = _3x3reduce1x1.Layer(NetworkLayers.Convolutional((3, 3), 20, ActivationFunctionType.ReLU));

                var stack = _1x1.DepthConcatenation(_3x3);
                stack.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid)); // Missing output node
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        [TestMethod]
        public void InitializationFail2()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));

                var split = conv1.TrainingBranch();
                split.Layer(NetworkLayers.Softmax(11)); // Output layer not matching the main output layer

                var fc1 = conv1.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        [TestMethod]
        public void InitializationFail3()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));

                var sum = _1x1.Sum(pool1); // Sum node inputs with different shapes
                var fc = sum.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc.Layer(NetworkLayers.Softmax(10));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        [TestMethod]
        public void InitializationFail4()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var split = root.TrainingBranch(); // Training branch starting from the input node
                split.Layer(NetworkLayers.Softmax(10));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3reduce1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3 = _3x3reduce1x1.Layer(NetworkLayers.Convolutional((3, 3), 20, ActivationFunctionType.ReLU));

                var stack = _1x1.DepthConcatenation(_3x3);
                var fc1 = stack.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        [TestMethod]
        public void InitializationFail5()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var conv1 = root.Layer(NetworkLayers.Convolutional((5, 5), 10, ActivationFunctionType.ReLU));
                var pool1 = conv1.Layer(NetworkLayers.Pooling(ActivationFunctionType.Sigmoid));

                var _1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3reduce1x1 = pool1.Layer(NetworkLayers.Convolutional((1, 1), 20, ActivationFunctionType.ReLU));
                var _3x3 = _3x3reduce1x1.Layer(NetworkLayers.Convolutional((3, 3), 20, ActivationFunctionType.ReLU));

                var stack = _1x1.DepthConcatenation(_3x3);
                var split = stack.TrainingBranch();
                var fct = split.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                var split2 = fct.TrainingBranch();
                split2.Layer(NetworkLayers.Softmax(10)); // Nested training branch
                fct.Layer(NetworkLayers.Softmax(10));
                var fc1 = stack.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fc1.Layer(NetworkLayers.Softmax(10));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        [TestMethod]
        public void InitializationFail6()
        {
            INeuralNetwork F() => NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(60, 60), root =>
            {
                var fc0 = root.Layer(NetworkLayers.FullyConnected(200, ActivationFunctionType.Sigmoid));

                var split = fc0.TrainingBranch();
                var fct = split.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                fct.Layer(NetworkLayers.Softmax(10));

                var fc1 = fc0.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                var sum = fct.Sum(fc1);
                sum.Layer(NetworkLayers.Softmax(10));
            });
            Assert.ThrowsException<ComputationGraphBuildException>(F);
        }

        #endregion

        #region Processing

        [TestMethod]
        public void ForwardTest1()
        {
            SequentialNetwork seq = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(12, 12),
                NetworkLayers.FullyConnected(20, ActivationFunctionType.Sigmoid, biasMode: BiasInitializationMode.Gaussian),
                NetworkLayers.Softmax(10)).To<INeuralNetwork, SequentialNetwork>();
            ComputationGraphNetwork graph = NetworkManager.NewGraph(seq.InputInfo, root =>
            {
                var fc = root.Layer(_ => seq.Layers[0].Clone());
                fc.Layer(_ => seq.Layers[1].Clone());
            }).To<INeuralNetwork, ComputationGraphNetwork>();
            float[,] x = new float[125, 12 * 12];
            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    x[i, j] = ThreadSafeRandom.NextFloat();
            float[,]
                ys = seq.Forward(x),
                yg = graph.Forward(x);
            Assert.IsTrue(ys.ContentEquals(yg));
        }

        [TestMethod]
        public void BackwardTest1()
        {
            SequentialNetwork seq = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(12, 12),
                NetworkLayers.FullyConnected(20, ActivationFunctionType.Sigmoid, biasMode: BiasInitializationMode.Gaussian),
                NetworkLayers.Softmax(10)).To<INeuralNetwork, SequentialNetwork>();
            ComputationGraphNetwork graph = NetworkManager.NewGraph(seq.InputInfo, root =>
            {
                var fc = root.Layer(_ => seq.Layers[0].Clone());
                fc.Layer(_ => seq.Layers[1].Clone());
            }).To<INeuralNetwork, ComputationGraphNetwork>();
            float[,]
                x = new float[125, 12 * 12],
                y = new float[125, 10];
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                    x[i, j] = ThreadSafeRandom.NextFloat();
                y[i, ThreadSafeRandom.NextInt(max: 10)] = 1;
            }
            SamplesBatch batch = new SamplesBatch(x, y);
            seq.Backpropagate(batch, 0, WeightsUpdaters.StochasticGradientDescent(TrainingAlgorithms.StochasticGradientDescent(0.1f)));
            graph.Backpropagate(batch, 0, WeightsUpdaters.StochasticGradientDescent(TrainingAlgorithms.StochasticGradientDescent(0.1f)));
            Assert.IsTrue(seq.Layers[0].Equals(graph.Layers[0]));
            Assert.IsTrue(seq.Layers[1].Equals(graph.Layers[1]));
        }

        #endregion

        #region Misc

        [TestMethod]
        public void JsonMetadataSerialization1()
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
            String json = network.SerializeMetadataAsJson();
            Assert.IsTrue(json != null);
        }

        [TestMethod]
        public void CloneTest()
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
            INeuralNetwork copy = network.Clone();
            Assert.IsTrue(network.Equals(copy));
            copy.Layers[0].To<INetworkLayer, ConvolutionalLayer>().Weights[0] += 0.1f;
            Assert.IsFalse(network.Equals(copy));
        }

        [TestMethod]
        public void SaveTest1()
        {
            INeuralNetwork network = NetworkManager.NewGraph(TensorInfo.Image<Alpha8>(28, 28), root =>
            {
                var fc1 = root.Layer(NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid));
                var fc2 = fc1.Layer(NetworkLayers.FullyConnected(40, ActivationFunctionType.Sigmoid));
                fc2.Layer(NetworkLayers.Softmax(10));
            });
            String path = Path.Combine(NetworkTest.DllPath, nameof(GraphNetworkTest));
            Directory.CreateDirectory(path);
            FileInfo file = new FileInfo(Path.Combine(path, "graph2.nnet"));
            network.Save(file);
            INeuralNetwork loaded = NetworkLoader.TryLoad(file, ExecutionModePreference.Cpu);
            Assert.IsTrue(loaded != null);
            Assert.IsTrue(network.Equals(loaded));
        }

        [TestMethod]
        public void SaveTest2()
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
            String path = Path.Combine(NetworkTest.DllPath, nameof(GraphNetworkTest));
            Directory.CreateDirectory(path);
            FileInfo file = new FileInfo(Path.Combine(path, "graph2.nnet"));
            network.Save(file);
            INeuralNetwork loaded = NetworkLoader.TryLoad(file, ExecutionModePreference.Cpu);
            Assert.IsTrue(loaded != null);
            Assert.IsTrue(network.Equals(loaded));
        }

        #endregion
    }
}
