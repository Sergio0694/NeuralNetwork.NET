using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Cuda.Convolution;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the <see cref="ConvolutionCudaExtensionsTest"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(ConvolutionCudaExtensionsTest))]
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public class ConvolutionCudaExtensionsTest
    {
        [TestMethod]
        public void ConvolutionForward()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(100, 784 * 3),
                kernels = random.NextXavierMatrix(10, 25 * 3),
                cpuResult = ConvolutionExtensions.Convolute(source, 3, kernels, 3, ConvolutionMode.Forward),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 3, kernels, 3, ConvolutionMode.Forward);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(100, 28 * 28 * 3),
                kernels = random.NextXavierMatrix(10, 24 * 24 * 10),
                cpuResult = ConvolutionExtensions.Convolute(source, 3, kernels, 10, ConvolutionMode.Backwards),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 3, kernels, 10, ConvolutionMode.Backwards);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void Pool2x2()
        {
            Random r = new Random();
            foreach (int size in new[] { 2, 4, 12, 400, 1000 })
            {
                float[,]
                    source = r.NextXavierMatrix(100, size * size * 17),
                    cpuResult = ConvolutionExtensions.Pool2x2(source, 17),
                    gpuResult = ConvolutionGpuExtensions.Pool2x2(source, 17);
                Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
            }
        }
    }
}
