using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
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
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
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
                    { 0.77, -0.11, 0.11, 0.33, 0.55, -0.11, 0.33 },
                    { -0.11, 1, -0.11, 0.33, -0.11, 0.11, -0.11 },
                    { 0.11, -0.11, 1, -0.33, 0.11, -0.11, 0.55 },
                    { 0.33, 0.33, -0.33, 0.55, -0.33, 0.33, 0.33 },
                    { 0.55, -0.11, 0.11, -0.33, 1, -0.11, 0.11 },
                    { -0.11, 0.11, -0.11, 0.33, -0.11, 1, -0.11 },
                    { 0.33, -0.11, 0.55, 0.33, 0.11, -0.11, 0.77 }
                },
                check = ConvolutionExtensions.ReLU(m);
            ConvolutionGpuExtensions.ReLU(m);
            Assert.IsTrue(m.ContentEquals(check));
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
                double[,]
                    source = new double[1, square],
                    m = r.NextXavierMatrix(size, size),
                    check = ConvolutionExtensions.Pool2x2(m);
                Buffer.BlockCopy(m, 0, source, 0, sizeof(double) * square);
                double[,]
                    m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                    unpacked = new double[half, half];
                Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(double) * halfSquare);
                Assert.IsTrue(unpacked.ContentEquals(check));
            }
        }

        [TestMethod]
        public void Pool2x2_2()
        {
            Random r = new Random();
            double[,]
                source = new double[1, 49],
                m = r.NextXavierMatrix(7, 7),
                check = ConvolutionExtensions.Pool2x2(m);
            Buffer.BlockCopy(m, 0, source, 0, sizeof(double) * 49);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked = new double[4, 4];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(double) * 16);
            Assert.IsTrue(unpacked.ContentEquals(check));
        }

        [TestMethod]
        public void Pool2x2_3()
        {
            Random r = new Random();
            double[,]
                source = new double[1, 9],
                m = r.NextXavierMatrix(3, 3),
                check = ConvolutionExtensions.Pool2x2(m);
            Buffer.BlockCopy(m, 0, source, 0, sizeof(double) * 9);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked = new double[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(double) * 4);
            Assert.IsTrue(unpacked.ContentEquals(check));
        }

        [TestMethod]
        public void Pool2x2_4()
        {
            Random r = new Random();
            double[,]
                source = new double[1, 18],
                m1 = r.NextXavierMatrix(3, 3),
                m2 = r.NextXavierMatrix(3, 3),
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2);
            Buffer.BlockCopy(m1, 0, source, 0, sizeof(double) * 9);
            Buffer.BlockCopy(m2, 0, source, sizeof(double) * 9, sizeof(double) * 9);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 2),
                unpacked1 = new double[2, 2],
                unpacked2 = new double[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, sizeof(double) * 4);
            Buffer.BlockCopy(m_gpu, sizeof(double) * 4, unpacked2, 0, sizeof(double) * 4);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Pool2x2_5()
        {
            Random r = new Random();
            double[,]
                source = new double[2, 16],
                m1 = r.NextXavierMatrix(4, 4),
                m2 = r.NextXavierMatrix(4, 4),
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2);
            Buffer.BlockCopy(m1, 0, source, 0, sizeof(double) * 16);
            Buffer.BlockCopy(m2, 0, source, sizeof(double) * 16, sizeof(double) * 16);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 1),
                unpacked1 = new double[2, 2],
                unpacked2 = new double[2, 2];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, sizeof(double) * 4);
            Buffer.BlockCopy(m_gpu, sizeof(double) * 4, unpacked2, 0, sizeof(double) * 4);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Pool2x2_6()
        {
            Random r = new Random();
            double[,]
                source = new double[2, 32],
                m1 = r.NextXavierMatrix(4, 4),
                m2 = r.NextXavierMatrix(4, 4),
                m3 = r.NextXavierMatrix(4, 4),
                m4 = r.NextXavierMatrix(4, 4),
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2),
                check3 = ConvolutionExtensions.Pool2x2(m3),
                check4 = ConvolutionExtensions.Pool2x2(m4);
            int size = 16, bytesize = sizeof(double) * size;
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            Buffer.BlockCopy(m3, 0, source, bytesize * 2, bytesize);
            Buffer.BlockCopy(m4, 0, source, bytesize * 3, bytesize);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 2),
                unpacked1 = new double[2, 2],
                unpacked2 = new double[2, 2],
                unpacked3 = new double[2, 2],
                unpacked4 = new double[2, 2];
            int halfBytesize = sizeof(double) * 4;
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
            double[,] source = new double[4, square * 3];
            for (int i = 0; i < 12; i++)
            {
                Buffer.BlockCopy(ms[i], 0, source, sizeof(double) * i * square, sizeof(double) * square);
            }
            double[,] m_gpu = ConvolutionGpuExtensions.Pool2x2(source, 3);
            int half = size / 2, halfSquare = half * half;
            var test = new double[4, halfSquare * 3];
            for (int i = 0; i < 12; i++)
            {
                Buffer.BlockCopy(checks[i], 0, test, sizeof(double) * i * halfSquare, sizeof(double) * halfSquare);
            }
            Assert.IsTrue(test.ContentEquals(m_gpu));
        }

        [TestMethod]
        public void Convolute3x3_1()
        {
            double[,]
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
                r = ConvolutionExtensions.Convolute3x3(m, k);
            const int size = 9;
            int
                square = size * size,
                inner = size - 2,
                innerSquare = inner * inner;
            double[,] source = new double[1, square];
            Buffer.BlockCopy(m, 0, source, 0, sizeof(double) * square);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k),
                unpacked = new double[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(double) * innerSquare);
            Assert.IsTrue(unpacked.ContentEquals(r));
        }

        [TestMethod]
        public void Convolute3x3_2()
        {
            // TODO
        }

        [TestMethod]
        public void Convolute3x3_3()
        {
            // TODO
        }
    }
}
