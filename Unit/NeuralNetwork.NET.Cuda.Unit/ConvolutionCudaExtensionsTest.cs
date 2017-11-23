using System;
using System.Diagnostics.CodeAnalysis;
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
        public void ConvolutionForward1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(100, 784),
                kernels = random.NextXavierMatrix(10, 25),
                cpuResult = ConvolutionExtensions.Convolute(source, 1, kernels, 1, ConvolutionMode.Forward),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 1, kernels, 1, ConvolutionMode.Forward);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionForward2()
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
        public void ConvolutionForward3()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(100, 1024 * 6),
                kernels = random.NextXavierMatrix(10, 9 * 6),
                cpuResult = ConvolutionExtensions.Convolute(source, 6, kernels, 6, ConvolutionMode.Forward),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 6, kernels, 6, ConvolutionMode.Forward);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(30, 28 * 28),
                kernels = random.NextXavierMatrix(1, 24 * 24 * 6),
                cpuResult = ConvolutionExtensions.Convolute(source, 1, kernels, 6, ConvolutionMode.Backwards),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 1, kernels, 6, ConvolutionMode.Backwards);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards2()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(25, 28 * 28 * 10),
                kernels = random.NextXavierMatrix(10, 24 * 24 * 3),
                cpuResult = ConvolutionExtensions.Convolute(source, 10, kernels, 3, ConvolutionMode.Backwards),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 10, kernels, 3, ConvolutionMode.Backwards);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards3()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(20, 28 * 28 * 4),
                kernels = random.NextXavierMatrix(4, 24 * 24 * 8),
                cpuResult = ConvolutionExtensions.Convolute(source, 4, kernels, 8, ConvolutionMode.Backwards),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 4, kernels, 8, ConvolutionMode.Backwards);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(30, 28 * 28 * 4),
                kernels = random.NextXavierMatrix(30, 24 * 24 * 7),
                cpuResult = ConvolutionExtensions.Convolute(source, 4, kernels, 7, ConvolutionMode.Gradient),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 4, kernels, 7, ConvolutionMode.Gradient);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient2()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(25, 28 * 28 * 10),
                kernels = random.NextXavierMatrix(25, 24 * 24 * 5),
                cpuResult = ConvolutionExtensions.Convolute(source, 10, kernels, 5, ConvolutionMode.Gradient),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 10, kernels, 5, ConvolutionMode.Gradient);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient3()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(10, 32 * 32 * 20),
                kernels = random.NextXavierMatrix(10, 24 * 24 * 10),
                cpuResult = ConvolutionExtensions.Convolute(source, 20, kernels, 10, ConvolutionMode.Gradient),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 20, kernels, 10, ConvolutionMode.Gradient);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }
    }
}
