using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A class with some test methods for the posted network class from python
    /// </summary>
    [TestClass]
    [TestCategory(nameof(NumpyNetworkTest))]
    public class NumpyNetworkTest
    {
        [TestMethod]
        public void TestNumpy1()
        {
            NumpyNetwork network = new NumpyNetwork(1, 2, 1)
            {
                weights =
                {
                    [0] = new[,] { { 1.34856747f }, { -1.16546082f } },
                    [1] = new[,] { { -0.73764399f, -0.69019199f } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544f }, { 0.66440039f } },
                    [1] = new[,] { { -0.01439235f } }
                }
            };
            float[,] value = network.feedforward(new[,] { { 1f } });
            Assert.IsTrue((value[0, 0] - 0.28743771f).Abs() < 0.1f);
        }

        [TestMethod]
        public void TestNumpy2()
        {
            NumpyNetwork network = new NumpyNetwork(1, 2, 1)
            {
                weights =
                {
                    [0] = new[,] { { 1.34856747f }, { -1.16546082f } },
                    [1] = new[,] { { -0.73764399f, -0.69019199f } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544f }, { 0.66440039f } },
                    [1] = new[,] { { -0.01439235f } }
                }
            };
            (float[][,] dJdb, float[][,] dJdw) = network.backprop(new[,] { { 1.2f } }, new[,] { { 1.2f } });
            Assert.IsTrue(dJdb.Length == 2 &&
                          dJdb[0].GetLength(0) == 2 && dJdb[0].GetLength(1) == 1 &&
                          dJdb[1].Length == 1);
            Assert.IsTrue(dJdw.Length == 2 &&
                          dJdw[0].GetLength(0) == 2 && dJdw[0].GetLength(1) == 1 &&
                          dJdw[1].GetLength(0) == 1 && dJdw[1].GetLength(1) == 2);
            Assert.IsTrue(dJdb[0][0, 0].EqualsWithDelta(0.01375305f, 1e-5f) &&
                          dJdb[0][1, 0].EqualsWithDelta(0.02834903f, 1e-5f) &&
                          dJdb[1][0, 0].EqualsWithDelta(-0.18744699f, 1e-5f) &&
                          dJdw[0][0, 0].EqualsWithDelta(0.01650366f, 1e-5f) &&
                          dJdw[0][1, 0].EqualsWithDelta(0.03401884f, 1e-5f) &&
                          dJdw[1][0, 0].EqualsWithDelta(-0.16645057f, 1e-5f) &&
                          dJdw[1][0, 1].EqualsWithDelta(-0.06078609f, 1e-5f));
        }

        private static (IReadOnlyList<(float[,], float[,])> TrainingData, IReadOnlyList<(float[,], float[,])> TestData) ParseMnistDataset()
        {
            const String TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";
            String TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";
            const String TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";
            const String TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";
            String
                code = Assembly.GetExecutingAssembly().Location,
                dll = Path.GetFullPath(code),
                root = Path.GetDirectoryName(dll),
                path = Path.Combine(root, "Assets");
            (float[,], float[,])[] ParseSamples(String valuePath, String labelsPath, int count)
            {
                (float[,], float[,])[] samples = new (float[,], float[,])[count];
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
                        float[,] sample = new float[784, 1];
                        for (int j = 0; j < 784; j++)
                        {
                            sample[j, 0] = temp[j] / 255f;
                        }

                        // Read the label
                        float[,] label = new float[10, 1];
                        int l = yGzip.ReadByte();
                        label[l, 0] = 1;

                        var tuple = (sample, label);
                        samples[i] = tuple;
                    }
                    return samples;
                }
            }
            return (ParseSamples(Path.Combine(path, TrainingSetValuesFilename), Path.Combine(path, TrainingSetLabelsFilename), 50_000),
                    ParseSamples(Path.Combine(path, TestSetValuesFilename), Path.Combine(path, TestSetLabelsFilename), 10_000));
        }

        [TestMethod]
        public void TrainingTest1()
        {
            (IReadOnlyList<(float[,], float[,])> TrainingData, 
             IReadOnlyList<(float[,], float[,])> TestData) data = ParseMnistDataset();
            NumpyNetwork net = new NumpyNetwork(784, 30, 10);
            (float[,], float)[] test = data.TestData.Select(t => (t.Item1, (float)t.Item2.Argmax())).ToArray();
            net.SGD(data.TrainingData, 1, 10, 3.0f, test);
            Assert.IsTrue(net.evaluate(test) > 8000);
        }
    }
}
