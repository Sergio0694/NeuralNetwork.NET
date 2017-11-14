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

        // 1-depth, 3*3 with 2*2 = 2*2 result
        [TestMethod]
        public void Convolution2DValid1()
        {
            float[,]
                l = { { 0, 1, 0, 2, 0, 1, 1, 1, 0 } },  // 3*3
                k = { { 1, 1, 0, 1 } };                 // 2*2
            float[,] result = l.Convolute(1, k, ConvolutionMode.Valid);
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
            float[,] result = l.Convolute(1, k, ConvolutionMode.Valid);
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
            float[,] result = l.Convolute(1, k, ConvolutionMode.Valid);
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
            float[,] result = l.Convolute(1, k, ConvolutionMode.Valid);
            float[,] expected = { { 2, 4, 6, 3 } };                                 // 2*2 result
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionBackupFull()
        {
            float[]
                l = { 0, 1, -1, 2 },
                k = { 1, 1, 0, 1 };
            float[] result = ConvolutionExtensions.convolute(l, 2, 2, k, 2, 2, ConvolutionMode.Full);
            float[] expected = { 0, 1, 1, -1, 1, 3, 0, -1, 2 };
            Assert.IsTrue(result.ContentEquals(expected));
        }

        [TestMethod]
        public void ConvolutionBackupValid()
        {
            float[]
                l = { 0, 1, 0, 2, 0, 1, 1, 1, 0 },
                k = { 1, 1, 0, 1 };
            float[] result = ConvolutionExtensions.convolute(l, 3, 3, k, 2, 2, ConvolutionMode.Valid);
            float[] expected = { 2, 2, 4, 1 };
            Assert.IsTrue(result.ContentEquals(expected));
        }
    }
}
