using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.Helpers;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;

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
        public void InPlaceSubtractAndHadamardProductWithActivationPrime()
        {
            Random r = new Random();
            float[,]
                m1 = r.NextGaussianMatrix(10, 10),
                m2 = r.NextGaussianMatrix(10, 10),
                m3 = r.NextGaussianMatrix(10, 10),
                backup = new float[10, 10];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            CostFunctions.QuadraticCostPrime(backup, m2, m3, ActivationFunctions.SigmoidPrime);
            CostFunctions.QuadraticCostPrime(m1, m2, m3, ActivationFunctions.SigmoidPrime);
            Assert.IsTrue(m1.ContentEquals(backup));
            m1 = r.NextGaussianMatrix(200, 200);
            m2 = r.NextGaussianMatrix(200, 200);
            m3 = r.NextGaussianMatrix(200, 200);
            backup = new float[200, 200];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(float) * m1.Length);
            CostFunctions.QuadraticCostPrime(backup, m2, m3, ActivationFunctions.SigmoidPrime);
            CostFunctions.QuadraticCostPrime(m1, m2, m3, ActivationFunctions.SigmoidPrime);
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
