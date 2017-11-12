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
        public void Convolution3x3_1()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, -1, -1, -1, -1, -1, -1, -1, -1 },
                    { -1, 1, -1, -1, -1, -1, -1, 1, -1 },
                    { -1, -1, 1, -1, -1, -1, 1, -1, -1 },
                    { -1, -1, -1, 1, -1, 1, -1, -1, -1 },
                    { -1, -1, -1, -1, 1, -1, -1, -1, -1 },
                    { -1, -1, -1, 1, -1, 1, -1, -1, -1 },
                    { -1, -1, 1, -1, -1, -1, 1, -1, -1 },
                    { -1, 1, -1, -1, -1, -1, -1, 1, -1 },
                    { -1, -1, -1, -1, -1, -1, -1, -1, -1 }
                },
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                r =
                {
                    { 0.77f, -0.11f, 0.11f, 0.33f, 0.55f, -0.11f, 0.33f },
                    { -0.11f, 1, -0.11f, 0.33f, -0.11f, 0.11f, -0.11f },
                    { 0.11f, -0.11f, 1, -0.33f, 0.11f, -0.11f, 0.55f },
                    { 0.33f, 0.33f, -0.33f, 0.55f, -0.33f, 0.33f, 0.33f },
                    { 0.55f, -0.11f, 0.11f, -0.33f, 1, -0.11f, 0.11f },
                    { -0.11f, 0.11f, -0.11f, 0.33f, -0.11f, 1, -0.11f },
                    { 0.33f, -0.11f, 0.55f, 0.33f, 0.11f, -0.11f, 0.77f }
                },
                t = m.Convolute3x3(k);
            t.Tweak(d => (float)Math.Truncate(d * 100) / 100f);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Convolution3x3_2()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                r =
                {
                    { 0.77f }
                },
                t = m.Convolute3x3(k);
            t.Tweak(d => (float)Math.Truncate(d * 100) / 100f);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Convolution3x3_3()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, -1, -1, -1, -1, -1 },
                    { -1, 1, -1, -1, -1, -1 },
                    { -1, -1, 1, -1, -1, -1 },
                    { -1, -1, -1, 1, -1, 1 },
                    { -1, -1, -1, -1, 1, -1 },
                    { -1, -1, -1, 1, -1, 1 }
                },
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                r =
                {
                    { 0.77f, -0.11f, 0.11f, 0.33f },
                    { -0.11f, 1, -0.11f, 0.33f },
                    { 0.11f, -0.11f, 1, -0.33f },
                    { 0.33f, 0.33f, -0.33f, 0.55f }
                },
                t = m.Convolute3x3(k);
            t.Tweak(d => (float)Math.Truncate(d * 100) / 100f);
            Assert.IsTrue(t.ContentEquals(r));
        }
    }
}
