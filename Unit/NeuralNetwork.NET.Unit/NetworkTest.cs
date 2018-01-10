using System;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A class with some test methods for the neural networks in the library
    /// </summary>
    [TestClass]
    [TestCategory(nameof(NetworkTest))]
    public class NetworkTest
    {
        // Gets the target assets subfolder
        [Pure, NotNull]
        private static String GetAssetsPath()
        {
            String
                code = Assembly.GetExecutingAssembly().Location,
                dll = Path.GetFullPath(code),
                root = Path.GetDirectoryName(dll),
                path = Path.Combine(root, "Assets");
            return path;
        }

        private static ((float[,] X, float[,] Y) TrainingData, (float[,] X, float[,] Y) TestData) ParseMnistDataset(int training = 50_000, int test = 10_000)
        {
            const String TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";
            const String TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";
            const String TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";
            const String TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";
            String path = GetAssetsPath();
            (float[,], float[,]) ParseSamples(String valuePath, String labelsPath, int count)
            {
                float[,] 
                    x = new float[count, 784],
                    y = new float[count, 10];
                using (FileStream
                    xStream = File.OpenRead(Path.Combine(path, valuePath)),
                    yStream = File.OpenRead(Path.Combine(path, labelsPath)))
                using (GZipStream
                    xGzip = new GZipStream(xStream, CompressionMode.Decompress),
                    yGzip = new GZipStream(yStream, CompressionMode.Decompress))
                {
                    xGzip.Read(new byte[16], 0, 16);
                    yGzip.Read(new byte[8], 0, 8);
                    for (int i = 0; i < count; i++)
                    {
                        // Read the image pixel values
                        byte[] temp = new byte[784];
                        xGzip.Read(temp, 0, 784);
                        float[] sample = new float[784];
                        for (int j = 0; j < 784; j++)
                        {
                            sample[j] = temp[j] / 255f;
                        }

                        // Read the label
                        float[,] label = new float[10, 1];
                        int l = yGzip.ReadByte();
                        label[l, 0] = 1;

                        // Copy to result matrices
                        Buffer.BlockCopy(sample, 0, x, sizeof(float) * i * 784, sizeof(float) * 784);
                        Buffer.BlockCopy(label, 0, y, sizeof(float) * i * 10, sizeof(float) * 10);
                    }
                    return (x, y);
                }
            }
            return (ParseSamples(Path.Combine(path, TrainingSetValuesFilename), Path.Combine(path, TrainingSetLabelsFilename), training),
                    ParseSamples(Path.Combine(path, TestSetValuesFilename), Path.Combine(path, TestSetLabelsFilename), test));
        }

        private static bool TestTrainingMethod(ITrainingAlgorithmInfo info)
        {
            (var trainingSet, var testSet) = ParseMnistDataset();
            BatchesCollection batches = BatchesCollection.From(trainingSet, 100);
            SequentialNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
                NetworkLayers.FullyConnected(100, ActivationFunctionType.Sigmoid),
                NetworkLayers.Softmax(10)).To<INeuralNetwork, SequentialNetwork>();
            TrainingSessionResult result = NetworkTrainer.TrainNetwork(network, batches, 1, 0, info, null, null, null, null, default);
            Assert.IsTrue(result.StopReason == TrainingStopReason.EpochsCompleted);
            (_, _, float accuracy) = network.Evaluate(testSet);
            if (accuracy < 80)
            {
                // Try again, just in case
                result = NetworkTrainer.TrainNetwork(network, batches, 1, 0, info, null, null, null, null, default);
                Assert.IsTrue(result.StopReason == TrainingStopReason.EpochsCompleted);
                (_, _, accuracy) = network.Evaluate(testSet);
            }
            return accuracy > 80;
        }

        [TestMethod]
        public void GradientDescentTest() => Assert.IsTrue(TestTrainingMethod(TrainingAlgorithms.StochasticGradientDescent()));

        [TestMethod]
        public void MomentumTest() => Assert.IsTrue(TestTrainingMethod(TrainingAlgorithms.Momentum()));

        [TestMethod]
        public void AdaGradTest() => Assert.IsTrue(TestTrainingMethod(TrainingAlgorithms.AdaGrad()));

        [TestMethod]
        public void AdaDeltaTest() => Assert.IsTrue(TestTrainingMethod(TrainingAlgorithms.AdaDelta()));

        [TestMethod]
        public void AdamTest() => Assert.IsTrue(TestTrainingMethod(TrainingAlgorithms.Adam()));

        [TestMethod]
        public void AdaMaxTest()
        {
            (var trainingSet, var testSet) = ParseMnistDataset(5000);
            BatchesCollection batches = BatchesCollection.From(trainingSet, 100);
            SequentialNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
                NetworkLayers.Convolutional((5, 5), 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
                NetworkLayers.Convolutional((3, 3), 40, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
                NetworkLayers.FullyConnected(125, ActivationFunctionType.LeCunTanh),
                NetworkLayers.Softmax(10)).To<INeuralNetwork, SequentialNetwork>();
            ITrainingAlgorithmInfo info = TrainingAlgorithms.AdaMax();
            TrainingSessionResult result = NetworkTrainer.TrainNetwork(network, batches, 1, 0, info, null, null, null, null, default);
            Assert.IsTrue(result.StopReason == TrainingStopReason.EpochsCompleted);
            (_, _, float accuracy) = network.Evaluate(testSet);
            Assert.IsTrue(accuracy > 80);
        }
    }
}
