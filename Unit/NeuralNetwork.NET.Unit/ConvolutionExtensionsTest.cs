using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="ConvolutionExtensions"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(ConvolutionExtensions))]
    public class ConvolutionExtensionsTest
    {
        [TestMethod]
        public void ReLU1()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, -0.1f, 2 },
                    { 1, 1, 2 },
                    { 0, -0.3f, 99 }
                },
                r =
                {
                    { 0, 0, 2 },
                    { 1, 1, 2 },
                    { 0, 0, 99 }
                },
                t = m.ReLU();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void ReLU2()
        {
            // Test values
            float[,]
                m =
                {
                    { 0.77f, -0.11f, 0.11f, 0.33f, 0.55f, -0.11f, 0.33f },
                    { -0.11f, 1, -0.11f, 0.33f, -0.11f, 0.11f, -0.11f },
                    { 0.11f, -0.11f, 1, -0.33f, 0.11f, -0.11f, 0.55f },
                    { 0.33f, 0.33f, -0.33f, 0.55f, -0.33f, 0.33f, 0.33f },
                    { 0.55f, -0.11f, 0.11f, -0.33f, 1, -0.11f, 0.11f },
                    { -0.11f, 0.11f, -0.11f, 0.33f, -0.11f, 1, -0.11f },
                    { 0.33f, -0.11f, 0.55f, 0.33f, 0.11f, -0.11f, 0.77f }
                },
                r =
                {
                    { 0.77f, 0, 0.11f, 0.33f, 0.55f, 0, 0.33f },
                    { 0, 1, 0, 0.33f, 0, 0.11f, 0 },
                    { 0.11f, 0, 1, 0, 0.11f, 0, 0.55f },
                    { 0.33f, 0.33f, 0, 0.55f, 0, 0.33f, 0.33f },
                    { 0.55f, 0, 0.11f, 0, 1, 0, 0.11f },
                    { 0, 0.11f, 0, 0.33f, 0, 1, 0 },
                    { 0.33f, 0, 0.55f, 0.33f, 0.11f, 0, 0.77f }
                },
                t = m.ReLU();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool2x2_1()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, 0, 1, 2 },
                    { 1, 1, 1, 1 },
                    { 0, -0.3f, -5, -0.5f },
                    { -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2 },
                    { 10, -0.5f }
                },
                t = m.Pool2x2();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool2x2_2()
        {
            // Test values
            float[,]
                m =
                {
                    { 0.77f, -0.11f, 0.11f, 0.33f, 0.55f, -0.11f, 0.33f },
                    { -0.11f, 1, -0.11f, 0.33f, -0.11f, 0.11f, -0.11f },
                    { 0.11f, -0.11f, 1, -0.33f, 0.11f, -0.11f, 0.55f },
                    { 0.33f, 0.33f, -0.33f, 0.55f, -0.33f, 0.33f, 0.33f },
                    { 0.55f, -0.11f, 0.11f, -0.33f, 1, -0.11f, 0.11f },
                    { -0.11f, 0.11f, -0.11f, 0.33f, -0.11f, 1, -0.11f },
                    { 0.33f, -0.11f, 0.55f, 0.33f, 0.11f, -0.11f, 0.77f }
                },
                r =
                {
                    { 1, 0.33f, 0.55f, 0.33f },
                    { 0.33f, 1, 0.33f, 0.55f },
                    { 0.55f, 0.33f, 1, 0.11f },
                    { 0.33f, 0.55f, 0.11f, 0.77f }
                },
                t = m.Pool2x2();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool2x2_3()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, 0 },
                    { 1, 1 },
                },
                r =
                {
                    { 1 }
                },
                t = m.Pool2x2();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Convolution1()
        {
            float[,]
                l1 =
                {
                    { 0, 0, 0, 1, 0 },
                    { 1, 2, 1, 2, 2 },
                    { 0, 2, 1, 2, 2 },
                    { 1, 0, 0, 0, 0 },
                    { 1, 1, 1, 0, 1 }
                },
                l2 =
                {
                    { 1, 1, 2, 0, 1 },
                    { 0, 1, 2, 1, 1 },
                    { 2, 0, 0, 0, 1 },
                    { 0, 2, 2, 1, 2 },
                    { 1, 0, 1, 0, 0 }
                },
                l3 =
                {
                    { 2, 1, 0, 1, 1 },
                    { 2, 2, 0, 2, 0 },
                    { 2, 1, 1, 0, 1 },
                    { 2, 1, 0, 2, 1 },
                    { 1, 2, 0, 0, 1 }
                },
                k1 =
                {
                    { 1, 1, -1 },
                    { 0, 1, -1 },
                    { 1, 1, -1 }
                },
                k2 =
                {
                    { 0, -1, -1 },
                    { -1, 0, -1 },
                    { 1, -1, -1 }
                },
                k3 =
                {
                    { 1, 1, 0 },
                    { -1, 1, 1 },
                    { -1, 1, 1 }
                };
            float[,]
                source = new float[1, 75],
                kernels = new float[1, 27];
            Buffer.BlockCopy(l1, 0, source, 0, sizeof(float) * 25);
            Buffer.BlockCopy(l2, 0, source, sizeof(float) * 25, sizeof(float) * 25);
            Buffer.BlockCopy(l3, 0, source, sizeof(float) * 50, sizeof(float) * 25);
            Buffer.BlockCopy(k1, 0, kernels, 0, sizeof(float) * 9);
            Buffer.BlockCopy(k2, 0, kernels, sizeof(float) * 9, sizeof(float) * 9);
            Buffer.BlockCopy(k3, 0, kernels, sizeof(float) * 18, sizeof(float) * 9);
            float[,] result = source.Convolute(3, kernels, ConvolutionMode.Valid);
            Assert.IsTrue(result.GetLength(0) == 1);
            Assert.IsTrue(result.GetLength(1) == 9);
            float[,] check = new float[3, 3];
            Buffer.BlockCopy(result, 0, check, 0, sizeof(float) * 9);
            float[,] expected =
            {
                { 2, -4, 0 },
                { -2, -1, 2 },
                { 3, 0, 2 }
            };
            Assert.IsTrue(check.ContentEquals(expected));
        }
    }
}
