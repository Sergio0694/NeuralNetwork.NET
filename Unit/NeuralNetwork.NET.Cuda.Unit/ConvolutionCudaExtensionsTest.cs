using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Cuda.Helpers;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the <see cref="ConvolutionCudaExtensionsTest"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(ConvolutionCudaExtensionsTest))]
    public class ConvolutionCudaExtensionsTest
    {
        [TestMethod]
        public void ReLU1()
        {
            // Test values
            double[,]
                m =
                {
                    { -1, -0.1, 2 },
                    { 1, 1, 2 },
                    { 0, -0.3, 99 }
                },
                r =
                {
                    { 0, 0, 2 },
                    { 1, 1, 2 },
                    { 0, 0, 99 }
                };
            ConvolutionGpuExtensions.ReLU(m);
            Assert.IsTrue(m.ContentEquals(r));
        }

        [TestMethod]
        public void ReLU2()
        {
            // Test values
            double[,]
                m =
                {
                    {0.77, -0.11, 0.11, 0.33, 0.55, -0.11, 0.33},
                    {-0.11, 1, -0.11, 0.33, -0.11, 0.11, -0.11},
                    {0.11, -0.11, 1, -0.33, 0.11, -0.11, 0.55},
                    {0.33, 0.33, -0.33, 0.55, -0.33, 0.33, 0.33},
                    {0.55, -0.11, 0.11, -0.33, 1, -0.11, 0.11},
                    {-0.11, 0.11, -0.11, 0.33, -0.11, 1, -0.11},
                    {0.33, -0.11, 0.55, 0.33, 0.11, -0.11, 0.77}
                },
                check = ConvolutionExtensions.ReLU(m);
            ConvolutionGpuExtensions.ReLU(m);
            Assert.IsTrue(m.ContentEquals(check));
        }

        /// <summary>
        /// Pool 2x2 test
        /// </summary>
        [TestMethod]
        public void Pool2x2()
        {
            // Test values
            double[,]
                m =
                {
                    { -1, 0, 1, 2 },
                    { 1, 1, 1, 1 },
                    { 0, -0.3, -5, -0.5 },
                    { -1, 10, -2, -1 }
                },
                r =
                {
                    { 1, 2 },
                    { 10, -0.5 }
                },
                t = m.Pool2x2();
            Assert.IsTrue(t.ContentEquals(r));
        }
    }
}
