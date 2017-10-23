using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Helpers;

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
        public void Multiply()
        {
            Random r = new Random();
            double[,]
                m1 = r.NextMatrix(7, 3),
                m2 = r.NextMatrix(3, 4),
                check = MatrixExtensions.Multiply(m1, m2);
            double[,] test = m1.Multiply(m2);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextMatrix(1500, 800);
            m2 = r.NextMatrix(800, 40);
            check = MatrixExtensions.Multiply(m1, m2);
            test = m1.Multiply(m2);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void MultiplyAndSigmoid()
        {
            Random r = new Random();
            double[,]
                m1 = r.NextMatrix(7, 3),
                m2 = r.NextMatrix(3, 4),
                check = MatrixExtensions.MultiplyAndSigmoid(m1, m2);
            double[,] test = m1.MultiplyAndSigmoid(m2);
            Assert.IsTrue(test.ContentEquals(check));
            m1 = r.NextMatrix(1500, 800);
            m2 = r.NextMatrix(800, 40);
            check = MatrixExtensions.MultiplyAndSigmoid(m1, m2);
            test = m1.MultiplyAndSigmoid(m2);
            Assert.IsTrue(test.ContentEquals(check));
        }

        [TestMethod]
        public void InPlaceSubtractAndHadamardProductWithSigmoidPrime()
        {
            Random r = new Random();
            double[,]
                m1 = r.NextMatrix(10, 10),
                m2 = r.NextMatrix(10, 10),
                m3 = r.NextMatrix(10, 10),
                backup = new double[10, 10];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(double) * m1.Length);
            MatrixExtensions.InPlaceSubtractAndHadamardProductWithSigmoidPrime(backup, m2, m3);
            m1.InPlaceSubtractAndHadamardProductWithSigmoidPrime(m2, m3);
            Assert.IsTrue(m1.ContentEquals(backup));
            m1 = r.NextMatrix(200, 200);
            m2 = r.NextMatrix(200, 200);
            m3 = r.NextMatrix(200, 200);
            backup = new double[200, 200];
            Buffer.BlockCopy(m1, 0, backup, 0, sizeof(double) * m1.Length);
            MatrixExtensions.InPlaceSubtractAndHadamardProductWithSigmoidPrime(backup, m2, m3);
            m1.InPlaceSubtractAndHadamardProductWithSigmoidPrime(m2, m3);
            Assert.IsTrue(m1.ContentEquals(backup));
        }
    }
}
