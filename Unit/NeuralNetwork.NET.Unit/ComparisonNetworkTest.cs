using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// A class with some test methods for the neural networks in the library
    /// </summary>
    [TestClass]
    [TestCategory(nameof(ComparisonNetworkTest))]
    public class ComparisonNetworkTest
    {
        [TestMethod]
        public void TestNumpy1()
        {
            // Initialization
            NumpyNetwork pyNet = new NumpyNetwork(1, 2, 1)
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
            NeuralNetwork dotNet = new NeuralNetwork(
                pyNet.weights.Select(MatrixExtensions.Transpose).ToArray(),
                pyNet.biases.Select(MatrixExtensions.Flatten).ToArray(), 
                pyNet.weights.Select(_ => ActivationFunctionType.Sigmoid).ToArray());

            // Tests
            double[,]
                pyResult = pyNet.feedforward(new[,] { { 1.2 } }),
                dotResult = dotNet.Forward(new[,] { { 1.2 } });
            Assert.IsTrue(pyResult[0, 0].EqualsWithDelta(dotResult[0, 0]));
            
            // Multiple samples
            double[,] samples = new Random().NextGaussianMatrix(80, 1);
            pyResult = pyNet.feedforward(samples.Transpose());
            dotResult = dotNet.Forward(samples);
            Assert.IsTrue(pyResult.Transpose().ContentEquals(dotResult));
        }

        [TestMethod]
        public void TestNumpy2()
        {
            NumpyNetwork pyNet = new NumpyNetwork(1, 2, 1)
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
            NeuralNetwork dotNet = new NeuralNetwork(
                pyNet.weights.Select(MatrixExtensions.Transpose).ToArray(),
                pyNet.biases.Select(MatrixExtensions.Flatten).ToArray(),
                pyNet.weights.Select(_ => ActivationFunctionType.Sigmoid).ToArray());

            // Tests
            (double[][,] dJdb, double[][,] dJdw) pyResult = pyNet.backprop(new[,] { { 1.2 } }, new[,] { { 1.0 } });
            double[] dotResult = dotNet.ComputeGradient(new[,] { { 1.2 } }, new[,] { { 1.0 } });
            double[] pyGradient = pyResult.dJdw.Zip(pyResult.dJdb, (w, b) => w.Flatten().Concat(b.Flatten()).ToArray()).Aggregate(new double[0], (s, v) => s.Concat(v).ToArray()).ToArray();
            Assert.IsTrue(dotResult.ContentEquals(pyGradient));

            // Additional Release/Debug test
            double[,]
                samples = { { 1.17 }, { 2.3 } },
                y = { { 1.0 }, { 0.5 } };
            pyResult = pyNet.backprop(samples.Transpose(), y.Transpose());
            dotResult = dotNet.ComputeGradient(samples, y);
            Assert.IsTrue((pyResult.dJdb[0][0, 0] + pyResult.dJdb[0][0, 1]).EqualsWithDelta(dotResult[2]));
            Assert.IsTrue((pyResult.dJdb[0][1, 0] + pyResult.dJdb[0][1, 1]).EqualsWithDelta(dotResult[3]));
            Assert.IsTrue((pyResult.dJdb[1][0, 0] + pyResult.dJdb[1][0, 1]).EqualsWithDelta(dotResult[6]));

            // Multiple samples
            Random r = new Random(7);
            samples = r.NextGaussianMatrix(160, 1);
            y = r.NextGaussianMatrix(160, 1);
            double[,] 
                pyF = pyNet.feedforward(samples.Transpose()),
                netF = dotNet.Forward(samples);
            Assert.IsTrue(pyF.Transpose().ContentEquals(netF));
            pyResult = pyNet.backprop(samples.Transpose(), y.Transpose());
            dotResult = dotNet.ComputeGradient(samples, y);
            double[][] dbs = pyResult.dJdb.Select(b =>
            {
                double[] db = new double[b.GetLength(0)];
                for (int i = 0; i < db.Length; i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                        db[i] += b[i, j];
                return db;
            }).ToArray();
            pyGradient = pyResult.dJdw.Zip(dbs, (w, b) => w.Flatten().Concat(b).ToArray()).Aggregate(new double[0], (s, v) => s.Concat(v).ToArray()).ToArray();
            Assert.IsTrue(dotResult.ContentEquals(pyGradient));
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
                (double[,], double[,])[] samples = new(double[,], double[,])[count];
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
    }
}
