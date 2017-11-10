using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Implementations.Misc;
using NeuralNetworkNET.Networks.PublicAPIs;

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
                    [0] = new[,] { { 1.34856747f }, { -1.16546082f } },
                    [1] = new[,] { { -0.73764399f, -0.69019199f } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544f }, { 0.66440039f } },
                    [1] = new[,] { { -0.01439235f } }
                }
            };
            NeuralNetwork dotNet = new NeuralNetwork(
                pyNet.weights.Select(MatrixExtensions.Transpose).ToArray(),
                pyNet.biases.Select(MatrixExtensions.Flatten).ToArray(), 
                pyNet.weights.Select(_ => ActivationFunctionType.Sigmoid).ToArray());

            // Tests
            float[,]
                pyResult = pyNet.feedforward(new[,] { { 1.2f } }),
                dotResult = dotNet.Forward(new[,] { { 1.2f } });
            Assert.IsTrue(pyResult[0, 0].EqualsWithDelta(dotResult[0, 0]));
            
            // Multiple samples
            float[,] samples = new Random().NextGaussianMatrix(80, 1);
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
                    [0] = new[,] { { 1.34856747f }, { -1.16546082f } },
                    [1] = new[,] { { -0.73764399f, -0.69019199f } }
                },
                biases =
                {
                    [0] = new[,] { { 0.45206544f }, { 0.66440039f } },
                    [1] = new[,] { { -0.01439235f } }
                }
            };
            NeuralNetwork dotNet = new NeuralNetwork(
                pyNet.weights.Select(MatrixExtensions.Transpose).ToArray(),
                pyNet.biases.Select(MatrixExtensions.Flatten).ToArray(),
                pyNet.weights.Select(_ => ActivationFunctionType.Sigmoid).ToArray());

            // Tests
            (float[][,] dJdb, float[][,] dJdw) pyResult = pyNet.backprop(new[,] { { 1.2f } }, new[,] { { 1.0f } });
            float[] dotResult = dotNet.Backpropagate(new[,] { { 1.2f } }, new[,] { { 1.0f } }).Flatten();
            float[] pyGradient = pyResult.dJdw.Zip(pyResult.dJdb, (w, b) => w.Flatten().Concat(b.Flatten()).ToArray()).Aggregate(new float[0], (s, v) => s.Concat(v).ToArray()).ToArray();
            Assert.IsTrue(dotResult.ContentEquals(pyGradient));

            // Additional Release/Debug test
            float[,]
                samples = { { 1.17f }, { 2.3f } },
                y = { { 1.0f }, { 0.5f } };
            pyResult = pyNet.backprop(samples.Transpose(), y.Transpose());
            dotResult = dotNet.Backpropagate(samples, y).Flatten();
            Assert.IsTrue((pyResult.dJdb[0][0, 0] + pyResult.dJdb[0][0, 1]).EqualsWithDelta(dotResult[2]));
            Assert.IsTrue((pyResult.dJdb[0][1, 0] + pyResult.dJdb[0][1, 1]).EqualsWithDelta(dotResult[3]));
            Assert.IsTrue((pyResult.dJdb[1][0, 0] + pyResult.dJdb[1][0, 1]).EqualsWithDelta(dotResult[6]));

            // Multiple samples
            Random r = new Random(7);
            samples = r.NextGaussianMatrix(160, 1);
            y = r.NextGaussianMatrix(160, 1);
            float[,] 
                pyF = pyNet.feedforward(samples.Transpose()),
                netF = dotNet.Forward(samples);
            Assert.IsTrue(pyF.Transpose().ContentEquals(netF));
            pyResult = pyNet.backprop(samples.Transpose(), y.Transpose());
            dotResult = dotNet.Backpropagate(samples, y).Flatten();
            float[][] dbs = pyResult.dJdb.Select(b =>
            {
                float[] db = new float[b.GetLength(0)];
                for (int i = 0; i < db.Length; i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                        db[i] += b[i, j];
                return db;
            }).ToArray();
            pyGradient = pyResult.dJdw.Zip(dbs, (w, b) => w.Flatten().Concat(b).ToArray()).Aggregate(new float[0], (s, v) => s.Concat(v).ToArray()).ToArray();
            Assert.IsTrue(dotResult.ContentEquals(pyGradient));
        }

        private static ((float[,] X, float[,] Y) TrainingData, (float[,] X, float[,] Y) TestData) ParseMnistDataset()
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
        public void GradientDescentTest()
        {
            (var trainingSet, var testSet) = ParseMnistDataset();
            NeuralNetwork network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(100, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            network.StochasticGradientDescent(trainingSet, 5, 100, null, null, 0.5f, 5);
            (_, float accuracy) = network.Evaluate(testSet);
            Console.WriteLine($"Accuracy: accuracy%");
            Assert.IsTrue(accuracy > 80);
        }
    }
}
