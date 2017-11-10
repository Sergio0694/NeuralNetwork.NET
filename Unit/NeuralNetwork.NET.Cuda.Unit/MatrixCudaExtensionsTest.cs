using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.APIs;
using NeuralNetworkNET.Cuda.Helpers;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;

#pragma warning disable 162

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the <see cref="MatrixCudaExtensionsTest"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(MatrixCudaExtensionsTest))]
    public class MatrixCudaExtensionsTest
    {
        [TestMethod]
        [SuppressMessage("ReSharper", "ReturnValueOfPureMethodIsNotUsed")]
        [SuppressMessage("ReSharper", "HeuristicUnreachableCode")]
        public void StopwatchTest()
        {
            // Helper
            void Benchmark(Action a1, Action a2, int iterations = 1)
            {
                while (iterations-- > 0)
                {
                    Stopwatch timer = new Stopwatch();
                    timer.Start();
                    a1();
                    timer.Stop();
                    var t1 = timer.ElapsedMilliseconds;
                    timer.Restart();
                    a2();
                    timer.Stop();
                    var t2 = timer.ElapsedMilliseconds;
                    Debug.WriteLine($"GPU: {t1}ms, CPU: {t2}ms");
                }
            }

            return;
            var network = NeuralNetwork.NewRandom(NetworkLayer.Inputs(200), NetworkLayer.FullyConnected(100, ActivationFunctionType.Sigmoid), NetworkLayer.FullyConnected(32, ActivationFunctionType.Sigmoid), NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            var r = new Random();
            var input = r.NextGaussianMatrix(2000, 200);
            var y = r.NextGaussianMatrix(2000, 10);
            Benchmark(() =>
            {
                NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
                network.Backpropagate(input, y);
            },
            () =>
            {
                NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Cpu;
                network.Backpropagate(input, y);
            }, 10);
        }

        [TestMethod]
        public void Transpose()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(7, 3),
                m2 = r.NextGaussianMatrix(25, 180),
                m3 = r.NextGaussianMatrix(1428, 3811);
            Assert.IsTrue(MatrixExtensions.Transpose(m1).ContentEquals(MatrixGpuExtensions.Transpose(m1)));
            Assert.IsTrue(MatrixExtensions.Transpose(m2).ContentEquals(MatrixGpuExtensions.Transpose(m2)));
            Assert.IsTrue(MatrixExtensions.Transpose(m3).ContentEquals(MatrixGpuExtensions.Transpose(m3)));
        }

        [TestMethod]
        public void Multiply()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(7, 3),
                m2 = r.NextGaussianMatrix(3, 4),
                check = MatrixExtensions.Multiply(m1, m2);
            float[,] test = MatrixGpuExtensions.Multiply(m1, m2);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextGaussianMatrix(1500, 800);
            m2 = r.NextGaussianMatrix(800, 40);
            check = MatrixExtensions.Multiply(m1, m2);
            test = MatrixGpuExtensions.Multiply(m1, m2);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void TransposeAndMultiply()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(5, 13),
                m2 = r.NextGaussianMatrix(5, 4),
                check = MatrixExtensions.Multiply(MatrixGpuExtensions.Transpose(m1), m2);
            float[,] test = m1.TransposeAndMultiply(m2);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextGaussianMatrix(800, 1500);
            m2 = r.NextGaussianMatrix(800, 40);
            check = MatrixExtensions.Multiply(MatrixGpuExtensions.Transpose(m1), m2);
            test = m1.TransposeAndMultiply(m2);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void MultiplyWithSum()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(13, 5),
                m2 = r.NextGaussianMatrix(5, 4);
            float[] v = Enumerable.Range(0, 4).Select(i => (float)i).ToArray();
            float[,] check = MatrixExtensions.MultiplyWithSum(m1, m2, v);
            float[,] test = MatrixGpuExtensions.MultiplyWithSum(m1, m2, v);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextGaussianMatrix(1500, 800);
            m2 = r.NextGaussianMatrix(800, 40);
            v = Enumerable.Range(0, 40).Select(i => (float)i).ToArray();
            check = MatrixExtensions.MultiplyWithSum(m1, m2, v);
            test = MatrixGpuExtensions.MultiplyWithSum(m1, m2, v);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void MultiplyWithSumAndActivation()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(13, 5),
                m2 = r.NextGaussianMatrix(5, 4);
            float[] v = Enumerable.Range(0, 4).Select(i => (float)i).ToArray();
            float[,] check = MatrixExtensions.MultiplyWithSumAndActivation(m1, m2, v, ActivationFunctions.Sigmoid);
            float[,] test = MatrixGpuExtensions.MultiplyWithSumAndActivation(m1, m2, v, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextGaussianMatrix(1500, 800);
            m2 = r.NextGaussianMatrix(800, 40);
            v = Enumerable.Range(0, 40).Select(i => (float)i).ToArray();
            check = MatrixExtensions.MultiplyWithSumAndActivation(m1, m2, v, ActivationFunctions.Sigmoid);
            test = MatrixGpuExtensions.MultiplyWithSumAndActivation(m1, m2, v, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void MultiplyAndActivation()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(7, 3),
                m2 = r.NextGaussianMatrix(3, 4),
                check = MatrixExtensions.MultiplyAndActivation(m1, m2, ActivationFunctions.Sigmoid);
            float[,] test = MatrixGpuExtensions.MultiplyAndActivation(m1, m2, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextGaussianMatrix(1500, 800);
            m2 = r.NextGaussianMatrix(800, 40);
            check = MatrixExtensions.MultiplyAndActivation(m1, m2, ActivationFunctions.Sigmoid);
            test = MatrixGpuExtensions.MultiplyAndActivation(m1, m2, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void Activation()
        {
            Random r = new Random();
            float[,]
                m = r.NextGaussianMatrix(20, 35),
                check = MatrixExtensions.Activation(m, ActivationFunctions.Sigmoid);
            float[,] test = MatrixGpuExtensions.Activation(m, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
            m = r.NextGaussianMatrix(1500, 800);
            check = MatrixExtensions.Activation(m, ActivationFunctions.Sigmoid);
            test = MatrixGpuExtensions.Activation(m, ActivationFunctions.Sigmoid);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void HalfSquaredDifference()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(7, 3),
                m2 = r.NextGaussianMatrix(7, 3);
            float
                check = MatrixExtensions.HalfSquaredDifference(m1, m2),
                test = MatrixGpuExtensions.HalfSquaredDifference(m1, m2);
            Assert.IsTrue(Math.Abs(check - test) < 0.0000001);
            m1 = r.NextGaussianMatrix(1500, 800);
            m2 = r.NextGaussianMatrix(1500, 800);
            check = MatrixExtensions.HalfSquaredDifference(m1, m2);
            test = MatrixGpuExtensions.HalfSquaredDifference(m1, m2);
            Assert.IsTrue(Math.Abs(check - test) < 0.0000001);
        }

        [TestMethod]
        public void InPlaceSubtractAndHadamardProductWithActivationPrime()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(10, 10),
                m2 = r.NextGaussianMatrix(10, 10),
                m3 = r.NextGaussianMatrix(10, 10),
                backup = new float[10, 10];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            MatrixExtensions.InPlaceSubtractAndHadamardProductWithActivationPrime(backup, m2, m3, ActivationFunctions.SigmoidPrime);
            MatrixGpuExtensions.InPlaceSubtractAndHadamardProductWithActivationPrime(m1, m2, m3, ActivationFunctions.SigmoidPrime);
            Assert.IsTrue(m1.ContentEquals(backup));
            m1 = r.NextGaussianMatrix(200, 200);
            m2 = r.NextGaussianMatrix(200, 200);
            m3 = r.NextGaussianMatrix(200, 200);
            backup = new float[200, 200];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            MatrixExtensions.InPlaceSubtractAndHadamardProductWithActivationPrime(backup, m2, m3, ActivationFunctions.SigmoidPrime);
            MatrixGpuExtensions.InPlaceSubtractAndHadamardProductWithActivationPrime(m1, m2, m3, ActivationFunctions.SigmoidPrime);
            Assert.IsTrue(m1.ContentEquals(backup));
        }

        [TestMethod]
        public void MultiplyAndInPlaceActivationPrimeAndHadamardProduct()
        {
            Random r = new Random();
            float[,]
                wt = r.NextGaussianMatrix(10, 10),
                m1 = r.NextGaussianMatrix(10, 10),
                m2 = r.NextGaussianMatrix(10, 10),
                backup = new float[10, 10];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            MatrixExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(backup, m2, wt, ActivationFunctions.SigmoidPrime);
            MatrixGpuExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(m1, m2, wt, ActivationFunctions.SigmoidPrime);
            Assert.IsTrue(m1.ContentEquals(backup));
            wt = r.NextGaussianMatrix(200, 200);
            m1 = r.NextGaussianMatrix(200, 200);
            m2 = r.NextGaussianMatrix(200, 200);
            backup = new float[200, 200];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            MatrixExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(backup, m2, wt, ActivationFunctions.SigmoidPrime);
            MatrixGpuExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(m1, m2, wt, ActivationFunctions.SigmoidPrime);
            Assert.IsTrue(m1.ContentEquals(backup));
        }
    }
}
