using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning.Data;

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
        private static String GetAssetsPath([NotNull] String folder)
        {
            String
                code = Assembly.GetExecutingAssembly().Location,
                dll = Path.GetFullPath(code),
                root = Path.GetDirectoryName(dll),
                path = Path.Combine(root, "Assets", folder);
            return path;
        }

        private static ((float[,] X, float[,] Y) TrainingData, (float[,] X, float[,] Y) TestData) ParseMnistDataset()
        {
            const String TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";
            String TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";
            const String TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";
            const String TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";
            String path = GetAssetsPath("MNIST");
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
            return (ParseSamples(Path.Combine(path, TrainingSetValuesFilename), Path.Combine(path, TrainingSetLabelsFilename), 50_000),
                    ParseSamples(Path.Combine(path, TestSetValuesFilename), Path.Combine(path, TestSetLabelsFilename), 10_000));
        }

        [TestMethod]
        public void ForwardTest1()
        {
            String path = GetAssetsPath("Networks");
            INeuralNetwork network = NeuralNetworkLoader.TryLoad(Path.Combine(path, "TestNetwork.nnet"));
            Assert.IsTrue(network != null);
            (_, var testSet) = ParseMnistDataset();
            float[,] yHat = network.Forward(testSet.X);
            Assert.IsTrue(yHat.GetUid() == -112610238);
        }

        [TestMethod]
        public void ForwardTest2()
        {
            String path = GetAssetsPath("Networks");
            INeuralNetwork network = NeuralNetworkLoader.TryLoad(Path.Combine(path, "TestNetwork.nnet"));
            Assert.IsTrue(network != null);
            (_, var testSet) = ParseMnistDataset();
            IReadOnlyList<(float[,] Z, float[,] A)> features = network.ExtractDeepFeatures(testSet.X);
            Assert.IsTrue(features[0].Z.GetUid() == -1663064875);
            Assert.IsTrue(features[0].A.GetUid() == 1115917777);
            Assert.IsTrue(features[1].Z.GetUid() == 1393557830);
            Assert.IsTrue(features[1].A.GetUid() == -112610238);
        }

        [TestMethod]
        public void GradientDescentTest1()
        {
            (var trainingSet, var testSet) = ParseMnistDataset();
            String path = GetAssetsPath("Networks");
            NeuralNetwork network = NeuralNetworkLoader.TryLoad(Path.Combine(path, "TestNetwork.nnet")) as NeuralNetwork;
            Assert.IsTrue(network != null);
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, 10);
            TrainingSessionResult result = NetworkTrainer.TrainNetwork(network, batches, 4, 0, TrainingAlgorithmsInfo.CreateForStochasticGradientDescent(), null, null, null, default);
            Assert.IsTrue(result.StopReason == TrainingStopReason.EpochsCompleted);
            (_, _, float accuracy) = network.Evaluate(testSet);
            Assert.IsTrue(accuracy > 80);
        }
    }
}
