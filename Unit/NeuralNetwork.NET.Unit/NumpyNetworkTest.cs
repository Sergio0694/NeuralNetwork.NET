using System;
using System.Collections.Generic;
using System.Diagnostics;
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
                    [0] = new[,] { { 1.34856747 }, { -1.16546082 } },
                    [1] = new[,] { { -0.73764399, -0.69019199 } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544 }, { 0.66440039 } },
                    [1] = new[,] { { -0.01439235 } }
                }
            };
            double[,] value = network.feedforward(new[,] { { 1d } });
            Assert.IsTrue((value[0, 0] - 0.28743771).Abs() < 0.1);
        }

        [TestMethod]
        public void TestNumpy2()
        {
            NumpyNetwork network = new NumpyNetwork(1, 2, 1)
            {
                weights =
                {
                    [0] = new[,] { { 1.34856747 }, { -1.16546082 } },
                    [1] = new[,] { { -0.73764399, -0.69019199 } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544 }, { 0.66440039 } },
                    [1] = new[,] { { -0.01439235 } }
                }
            };
            (double[][,] dJdb, double[][,] dJdw) = network.backprop(new[,] { { 1.2 } }, new[,] { { 1.2 } });
            Assert.IsTrue(dJdb.Length == 2 &&
                          dJdb[0].GetLength(0) == 2 && dJdb[0].GetLength(1) == 1 &&
                          dJdb[1].Length == 1);
            Assert.IsTrue(dJdw.Length == 2 &&
                          dJdw[0].GetLength(0) == 2 && dJdw[0].GetLength(1) == 1 &&
                          dJdw[1].GetLength(0) == 1 && dJdw[1].GetLength(1) == 2);
            Assert.IsTrue(dJdb[0][0, 0].EqualsWithDelta(0.01375305, 1e-5) &&
                          dJdb[0][1, 0].EqualsWithDelta(0.02834903, 1e-5) &&
                          dJdb[1][0, 0].EqualsWithDelta(-0.18744699, 1e-5) &&
                          dJdw[0][0, 0].EqualsWithDelta(0.01650366, 1e-5) &&
                          dJdw[0][1, 0].EqualsWithDelta(0.03401884, 1e-5) &&
                          dJdw[1][0, 0].EqualsWithDelta(-0.16645057, 1e-5) &&
                          dJdw[1][0, 1].EqualsWithDelta(-0.06078609, 1e-5));
        }

        private static (IReadOnlyList<(double[,], double[,])> TrainingData, IReadOnlyList<(double[,], double[,])> TestData) ParseMnistDataset()
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
            (double[,], double[,])[] ParseSamples(String valuePath, String labelsPath, int count)
            {
                (double[,], double[,])[] samples = new (double[,], double[,])[count];
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
                        double[,] sample = new double[784, 1];
                        for (int j = 0; j < 784; j++)
                        {
                            sample[j, 0] = temp[j] / 255d;
                        }

                        // Read the label
                        double[,] label = new double[10, 1];
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
            var data = ParseMnistDataset();
            Console.WriteLine("Dataset PARSED");
            Debug.WriteLine("Dataset PARSED");
            var net = new NumpyNetwork(784, 30, 10);
            var test = data.TestData.Select(t => (t.Item1, (double)t.Item2.Argmax())).ToArray();
            net.SGD(data.TrainingData, 1, 10, 3.0, test);
            Assert.IsTrue(net.evaluate(test) > 9000);
        }
    }
}
