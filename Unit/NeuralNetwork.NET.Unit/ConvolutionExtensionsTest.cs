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
        public void Pool1()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, -0.5f, -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2, 10, -0.5f }
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool2()
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
        public void Pool3()
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
        public void Pool4()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, -0.5f, -1, 10, -2, -1 },
                    { -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, 1.2f, -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2, 10, -0.5f },
                    { 1, 2, 10, 1.2f },
                },
                t = m.Pool2x2(1);
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void Pool5()
        {
            // Test values
            float[,]
                m =
                {
                    { -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, -0.5f, -1, 10, -2, -1, -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, 1.2f, -1, 10, -2, -1 },
                    { -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, 1.2f, -1, 10, -2, -1, -1, 0, 1, 2, 1, 1, 1, 1, 0, -0.3f, -5, 1.45f, -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2, 10, -0.5f, 1, 2, 10, 1.2f },
                    { 1, 2, 10, 1.2f, 1, 2, 10, 1.45f },
                },
                t = m.Pool2x2(2);
            Assert.IsTrue(t.ContentEquals(r));
        }

        // 1-depth, 3*3 with 2*2 = 2*2 result
        [TestMethod]
        public void Convolution2DValid1()
        {
            float[,]
                l = { { 0, 1, 0, 2, 0, 1, 1, 1, 0 } },  // 3*3
                k = { { 1, 1, 0, 1 } };                 // 2*2
            float[,] result = l.Convolute(1, k, 1, ConvolutionMode.Forward);
            float[,] expected = { { 2, 2, 4, 1 } };     // 2*2 
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 1-depth, 2 sample 3*3 with 2*2 = 2 sample 2*2 result
        [TestMethod]
        public void Convolution2DValid2()
        {
            float[,]
                l = { { 0, 1, 0, 2, 0, 1, 1, 1, 0 }, { 0, 1, 0, 2, 0, 1, 1, 1, 0 } },   // 2 sample, 3*3
                k = { { 1, 1, 0, 1 } };                                                 // 2*2
            float[,] result = l.Convolute(1, k, 1, ConvolutionMode.Forward);
            float[,] expected = { { 2, 2, 4, 1 }, { 2, 2, 4, 1 } };                     // 2 sample, 2*2
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 1-depth, 3*3 with 2 kernels 2*2 = 2-depth 2*2 result
        [TestMethod]
        public void Convolution2DValid3()
        {
            float[,]
                l = { { 0, 1, 0, 2, 0, 1, 1, 1, 0 } },              // 3*3
                k = { { 1, 1, 0, 1 }, { 0, 1, 2, 0 } };             // 2 kernels 2*2
            float[,] result = l.Convolute(1, k, 1, ConvolutionMode.Forward);
            float[,] expected = { { 2, 2, 4, 1, 4, 0, 1, 3 } };     // 2-depth, 2*2 result
            Assert.IsTrue(result.ContentEquals(expected));
        }

        // 2-depth, 3*3 with 2-depth kernel = 2*2 result
        [TestMethod]
        public void Convolution2DValid4()
        {
            float[,]
                l = { { 0, 1, 0, 2, 0, 1, 1, 1, 0, 1, 0, 0, 0, 2, 1, 0, 1, 1 } },   // 2-depth 3*3
                k = { { 1, 1, 0, 1, 0, 1, 1, 0 } };                                 // 2-depth 2*2 kernel
            float[,] result = l.Convolute(2, k, 2, ConvolutionMode.Forward);
            float[,] expected = { { 2, 4, 6, 3 } };                                 // 2*2 result
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionFull1()
        {
            float[,]
                l = { { 0, 1, -1, 2 } },
                k = { { 1, 1, 0, 1 } };
            float[,] result = l.Convolute(1, k, 1, ConvolutionMode.Backwards);
            float[,] expected = { { 0, 1, 1, -1, 1, 3, 0, -1, 2 } };
            Assert.IsTrue(result.ContentEquals(expected));
        }
    }
}
