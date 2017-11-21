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
        public void ConvolutionForward1()
        {
            Random random = new Random();
            float[,]
                source = random.NextXavierMatrix(10, 784),
                kernels = random.NextXavierMatrix(2, 25),
                cpuResult = ConvolutionExtensions.Convolute(source, 1, kernels, 1, ConvolutionMode.Forward),
                gpuResult = ConvolutionGpuExtensions.Convolute(source, 1, kernels, 1, ConvolutionMode.Forward);
            Assert.IsTrue(cpuResult.ContentEquals(gpuResult));
        }

        [TestMethod]
        public void Pool2x2_1()
        {
            Random r = new Random();
            foreach (int size in new[] { 2, 4, 12, 400, 1000 })
            {
                int
                    square = size * size,
                    half = size / 2,
                    halfSquare = half * half;
                float[,]
                    source = new float[1, square],
                    m = r.NextXavierMatrix(size, size),
                    check = ConvolutionExtensions.Pool2x2(m, 1);
                Buffer.BlockCopy(m, 0, source, 0, sizeof(float) * square);
                float[,]
                    m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                    unpacked = new float[half, half];
                Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(float) * halfSquare);
                Assert.IsTrue(unpacked.ContentEquals(check));
            }
        }

        [TestMethod]
        public void Pool2x2_2()
        {
            Random r = new Random();
            float[,]
                source = new float[1, 49],
                m = r.NextXavierMatrix(7, 7),
                check = ConvolutionExtensions.Pool2x2(m, 1);
            Buffer.BlockCopy(m, 0, source, 0, sizeof(float) * 49);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked = new float[4, 4];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(float) * 16);
            Assert.IsTrue(unpacked.ContentEquals(check));
        }

        [TestMethod]
        public void Pool2x2_3()
        {
            Random r = new Random();
            float[,]
                source = new float[1, 9],
                m = r.NextXavierMatrix(3, 3),
                check = ConvolutionExtensions.Pool2x2(m, 1);
            Buffer.BlockCopy(m, 0, source, 0, sizeof(float) * 9);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked = new float[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(float) * 4);
            Assert.IsTrue(unpacked.ContentEquals(check));
        }

        [TestMethod]
        public void Pool2x2_4()
        {
            Random r = new Random();
            float[,]
                source = new float[1, 18],
                m1 = r.NextXavierMatrix(3, 3),
                m2 = r.NextXavierMatrix(3, 3),
                check1 = ConvolutionExtensions.Pool2x2(m1, 2),
                check2 = ConvolutionExtensions.Pool2x2(m2, 2);
            Buffer.BlockCopy(m1, 0, source, 0, sizeof(float) * 9);
            Buffer.BlockCopy(m2, 0, source, sizeof(float) * 9, sizeof(float) * 9);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 2),
                unpacked1 = new float[2, 2],
                unpacked2 = new float[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, sizeof(float) * 4);
            Buffer.BlockCopy(m_gpu, sizeof(float) * 4, unpacked2, 0, sizeof(float) * 4);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Pool2x2_5()
        {
            Random r = new Random();
            float[,]
                source = new float[2, 16],
                m1 = r.NextXavierMatrix(4, 4),
                m2 = r.NextXavierMatrix(4, 4),
                check1 = ConvolutionExtensions.Pool2x2(m1, 1),
                check2 = ConvolutionExtensions.Pool2x2(m2, 1);
            Buffer.BlockCopy(m1, 0, source, 0, sizeof(float) * 16);
            Buffer.BlockCopy(m2, 0, source, sizeof(float) * 16, sizeof(float) * 16);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked1 = new float[2, 2],
                unpacked2 = new float[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, sizeof(float) * 4);
            Buffer.BlockCopy(m_gpu, sizeof(float) * 4, unpacked2, 0, sizeof(float) * 4);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Pool2x2_6()
        {
            Random r = new Random();
            float[,]
                source = new float[2, 32],
                m1 = r.NextXavierMatrix(4, 4),
                m2 = r.NextXavierMatrix(4, 4),
                m3 = r.NextXavierMatrix(4, 4),
                m4 = r.NextXavierMatrix(4, 4),
                check1 = ConvolutionExtensions.Pool2x2(m1, 2),
                check2 = ConvolutionExtensions.Pool2x2(m2, 2),
                check3 = ConvolutionExtensions.Pool2x2(m3, 2),
                check4 = ConvolutionExtensions.Pool2x2(m4, 2);
            int size = 16, bytesize = sizeof(float) * size;
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            Buffer.BlockCopy(m3, 0, source, bytesize * 2, bytesize);
            Buffer.BlockCopy(m4, 0, source, bytesize * 3, bytesize);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 2),
                unpacked1 = new float[2, 2],
                unpacked2 = new float[2, 2],
                unpacked3 = new float[2, 2],
                unpacked4 = new float[2, 2];
            int halfBytesize = sizeof(float) * 4;
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, halfBytesize);
            Buffer.BlockCopy(m_gpu, halfBytesize, unpacked2, 0, halfBytesize);
            Buffer.BlockCopy(m_gpu, halfBytesize * 2, unpacked3, 0, halfBytesize);
            Buffer.BlockCopy(m_gpu, halfBytesize * 3, unpacked4, 0, halfBytesize);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
            Assert.IsTrue(unpacked3.ContentEquals(check3));
            Assert.IsTrue(unpacked4.ContentEquals(check4));
        }

        [TestMethod]
        public void Pool2x2_7()
        {
            Random r = new Random();
            const int size = 40;
            int square = size * size;
            var ms = Enumerable.Range(0, 12).Select(_ => r.NextXavierMatrix(size, size)).ToArray();
            var checks = ms.Select(ConvolutionExtensions.Pool2x2).ToArray();
            float[,] source = new float[4, square * 3];
            for (int i = 0; i < 12; i++)
            {
                Buffer.BlockCopy(ms[i], 0, source, sizeof(float) * i * square, sizeof(float) * square);
            }
            float[,] m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 3);
            int half = size / 2, halfSquare = half * half;
            var test = new float[4, halfSquare * 3];
            for (int i = 0; i < 12; i++)
            {
                Buffer.BlockCopy(checks[i], 0, test, sizeof(float) * i * halfSquare, sizeof(float) * halfSquare);
            }
            Assert.IsTrue(test.ContentEquals(m_gpu));
        }
    }
}
