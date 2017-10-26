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
        /// <summary>
        /// ReLU test
        /// </summary>
        [TestMethod]
        public void ReLU()
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
                },
                t = m.ReLU();
            Assert.IsTrue(t.ContentEquals(r));
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
