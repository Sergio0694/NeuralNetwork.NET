using System;
using System.Diagnostics.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.Helpers;
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
            float[] biases = random.NextGaussianVector(10);
            float[,]
                source = random.NextXavierMatrix(100, 784),
                kernels = random.NextXavierMatrix(10, 25),
                cpuResult = ConvolutionExtensions.ConvoluteForward(source, (28, 28, 1), kernels, (5, 5, 1), biases),
                gpuResult = ConvolutionGpuExtensions.ConvoluteForward(source, (28, 28, 1), kernels, (5, 5, 1), biases);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionForward2()
        {
            Random random = new Random();
            float[] biases = random.NextGaussianVector(10);
            float[,]
                source = random.NextXavierMatrix(100, 784 * 3),
                kernels = random.NextXavierMatrix(10, 25 * 3),
                cpuResult = ConvolutionExtensions.ConvoluteForward(source, (28, 28, 3), kernels, (5, 5, 3), biases),
                gpuResult = ConvolutionGpuExtensions.ConvoluteForward(source, (28, 28, 3), kernels, (5, 5, 3), biases);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionForward3()
        {
            Random random = new Random();
            float[] biases = random.NextGaussianVector(10);
            float[,]
                source = random.NextXavierMatrix(100, 1024 * 6),
                kernels = random.NextXavierMatrix(10, 9 * 6),
                cpuResult = ConvolutionExtensions.ConvoluteForward(source, (32, 32, 6), kernels, (3, 3, 6), biases),
                gpuResult = ConvolutionGpuExtensions.ConvoluteForward(source, (32, 32, 6), kernels, (3, 3, 6), biases);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(30, 28 * 28),
                kernels = random.NextXavierMatrix(1, 24 * 24 * 6),
                cpuResult = ConvolutionExtensions.ConvoluteBackwards(source, (28, 28, 1), kernels, (24, 24, 6)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteBackwards(source, (28, 28, 1), kernels, (24, 24, 6));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards2()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(25, 28 * 28 * 10),
                kernels = random.NextXavierMatrix(10, 24 * 24 * 3),
                cpuResult = ConvolutionExtensions.ConvoluteBackwards(source, (28, 28, 10), kernels, (24, 24, 3)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteBackwards(source, (28, 28, 10), kernels, (24, 24, 3));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionBackwards3()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(20, 28 * 28 * 4),
                kernels = random.NextXavierMatrix(4, 24 * 24 * 8),
                cpuResult = ConvolutionExtensions.ConvoluteBackwards(source, (28, 28, 4), kernels, (24, 24, 8)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteBackwards(source, (28, 28, 4), kernels, (24, 24, 8));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(30, 28 * 28 * 4),
                kernels = random.NextXavierMatrix(30, 24 * 24 * 7),
                cpuResult = ConvolutionExtensions.ConvoluteGradient(source, (28, 28, 4), kernels, (24, 24, 7)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteGradient(source, (28, 28, 4), kernels, (24, 24, 7));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient2()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(25, 28 * 28 * 10),
                kernels = random.NextXavierMatrix(25, 24 * 24 * 5),
                cpuResult = ConvolutionExtensions.ConvoluteGradient(source, (28, 28, 10), kernels, (24, 24, 5)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteGradient(source, (28, 28, 10), kernels, (24, 24, 5));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }

        [TestMethod]
        public void ConvolutionGradient3()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(10, 32 * 32 * 20),
                kernels = random.NextXavierMatrix(10, 24 * 24 * 10),
                cpuResult = ConvolutionExtensions.ConvoluteGradient(source, (32, 32, 20), kernels, (24, 24, 10)),
                gpuResult = ConvolutionGpuExtensions.ConvoluteGradient(source, (32, 32, 20), kernels, (24, 24, 10));
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult, 1e-4f));
        }
    }
}
