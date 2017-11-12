using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Convolution.Operations;
using NeuralNetworkNET.Cuda.APIs;
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
                };
            ConvolutionGpuExtensions.ReLU(m);
            Assert.IsTrue(m.ContentEquals(r));
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
                float[,]
                    source = new float[1, square],
                    m = r.NextXavierMatrix(size, size),
                    check = ConvolutionExtensions.Pool2x2(m);
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
                check = ConvolutionExtensions.Pool2x2(m);
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
                check = ConvolutionExtensions.Pool2x2(m);
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
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2);
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
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2);
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
                check1 = ConvolutionExtensions.Pool2x2(m1),
                check2 = ConvolutionExtensions.Pool2x2(m2),
                check3 = ConvolutionExtensions.Pool2x2(m3),
                check4 = ConvolutionExtensions.Pool2x2(m4);
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

        [TestMethod]
        public void Convolute3x3_1()
        {
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
                r = ConvolutionExtensions.Convolute3x3(m, k);
            const int size = 3;
            int square = size * size;
            float[,] source = new float[1, square];
            Buffer.BlockCopy(m, 0, source, 0, sizeof(float) * square);
            float[][,] m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k);
            float[,] unpacked = m_gpu.MergeRows();
            Assert.IsTrue(unpacked.ContentEquals(r));
        }

        [TestMethod]
        public void Convolute3x3_2()
        {
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
                r = ConvolutionExtensions.Convolute3x3(m, k);
            const int size = 9;
            int
                square = size * size,
                inner = size - 2,
                innerSquare = inner * inner;
            float[,] source = new float[1, square];
            Buffer.BlockCopy(m, 0, source, 0, sizeof(float) * square);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k).MergeRows(),
                unpacked = new float[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(float) * innerSquare);
            Assert.IsTrue(unpacked.ContentEquals(r));
        }

        [TestMethod]
        public void Convolute3x3_3()
        {
            Random r = new Random();
            int 
                size = 3, 
                area = size * size, 
                bytesize = sizeof(float) * area,
                inner = size - 2,
                innerBytesize = sizeof(float) * inner;
            float[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new float[1, area * 2],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new float[inner, inner],
                unpacked2 = new float[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize, unpacked2, 0, innerBytesize);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Convolute3x3_4()
        {
            Random r = new Random();
            int
                size = 9,
                area = size * size,
                bytesize = sizeof(float) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(float) * innerArea;
            float[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new float[2, area],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k).MergeRows(),
                unpacked1 = new float[inner, inner],
                unpacked2 = new float[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize, unpacked2, 0, innerBytesize);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
        }

        [TestMethod]
        public void Convolute3x3_5()
        {
            Random r = new Random();
            int
                size = 9,
                area = size * size,
                bytesize = sizeof(float) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(float) * innerArea;
            float[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new float[2, area * 2],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                m3 = r.NextXavierMatrix(size, size),
                m4 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k),
                check3 = ConvolutionExtensions.Convolute3x3(m3, k),
                check4 = ConvolutionExtensions.Convolute3x3(m4, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            Buffer.BlockCopy(m3, 0, source, bytesize * 2, bytesize);
            Buffer.BlockCopy(m4, 0, source, bytesize * 3, bytesize);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new float[inner, inner],
                unpacked2 = new float[inner, inner],
                unpacked3 = new float[inner, inner],
                unpacked4 = new float[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize, unpacked2, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize * 2, unpacked3, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize * 3, unpacked4, 0, innerBytesize);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
            Assert.IsTrue(unpacked3.ContentEquals(check3));
            Assert.IsTrue(unpacked4.ContentEquals(check4));
        }

        [TestMethod]
        public void Convolute3x3_6()
        {
            Random r = new Random();
            int
                size = 9,
                area = size * size,
                bytesize = sizeof(float) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(float) * innerArea;
            float[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new float[2, area * 2],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                m3 = r.NextXavierMatrix(size, size),
                m4 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k),
                check3 = ConvolutionExtensions.Convolute3x3(m3, k),
                check4 = ConvolutionExtensions.Convolute3x3(m4, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            Buffer.BlockCopy(m3, 0, source, bytesize * 2, bytesize);
            Buffer.BlockCopy(m4, 0, source, bytesize * 3, bytesize);
            float[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new float[inner, inner],
                unpacked2 = new float[inner, inner],
                unpacked3 = new float[inner, inner],
                unpacked4 = new float[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked1, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize, unpacked2, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize * 2, unpacked3, 0, innerBytesize);
            Buffer.BlockCopy(m_gpu, innerBytesize * 3, unpacked4, 0, innerBytesize);
            Assert.IsTrue(unpacked1.ContentEquals(check1));
            Assert.IsTrue(unpacked2.ContentEquals(check2));
            Assert.IsTrue(unpacked3.ContentEquals(check3));
            Assert.IsTrue(unpacked4.ContentEquals(check4));
        }

        [TestMethod]
        public void PipelineTest1()
        {
            Random r = new Random();
            float[,] dataset = r.NextXavierMatrix(1, 9);
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                ConvolutionOperation.Convolution3x3(
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss));
            float[,] result_cpu = pipeline.Process(dataset);
            float[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.TopLeftEmboss).MergeRows();
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }

        [TestMethod]
        public void PipelineTest2()
        {
            Random r = new Random();
            float[,] dataset = r.NextXavierMatrix(1, 9);
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                ConvolutionOperation.Convolution3x3(
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                ConvolutionOperation.ReLU);
            float[,] result_cpu = pipeline.Process(dataset);
            float[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.TopLeftEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(result_gpu);
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }

        [TestMethod]
        public void PipelineTest3()
        {
            Random r = new Random();
            float[,] dataset = r.NextXavierMatrix(2, 9);
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                ConvolutionOperation.Convolution3x3(
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                ConvolutionOperation.ReLU);
            float[,] result_cpu = pipeline.Process(dataset);
            float[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.TopLeftEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(result_gpu);
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }

        [TestMethod]
        public void PipelineTest4()
        {
            Random r = new Random();
            float[,] dataset = r.NextXavierMatrix(10, 81);
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 9*9*1 pixels >> 7*7*3
                ConvolutionOperation.Convolution3x3(
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                ConvolutionOperation.ReLU,
                ConvolutionOperation.Pool2x2); // 7*7*3 >> 4*4*3
            float[,] result_cpu = pipeline.Process(dataset);
            float[,] p1 = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.TopLeftEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(p1);
            float[,] result_gpu = ConvolutionGpuExtensions.Pool2x2(p1, 3);
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }

        [TestMethod]
        public void PipelineTest5()
        {
            // Shared pipeline
            ConvolutionPipeline pipeline = new ConvolutionPipeline(
            ConvolutionOperation.Convolution3x3( // 10 kernels, 28*28*1 pixels >> 26*26*6
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen),
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2, // 26*26*6 >> 13*13*6,
            ConvolutionOperation.Convolution3x3(
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel),// 13*13*10 >> 11*11*24
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2, // 11*11*24 >> 6*6*24,
            ConvolutionOperation.Convolution3x3(
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline), // 6*6*24 >> 4*4*120
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2, // 4*4*120 >> 2*2*120
            ConvolutionOperation.Pool2x2); // 2*2*120 >> 1*1*120

            // Setup
            Random r = new Random();
            float[,] dataset = r.NextXavierMatrix(10000, 784);
            Stopwatch timer = new Stopwatch();
            timer.Start();

            // GPU
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            float[,] result_gpu = pipeline.Process(dataset);
            timer.Stop();
            var t1 = timer.ElapsedMilliseconds;

            // CPU
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Cpu;
            timer.Restart();
            float[,] result_cpu = pipeline.Process(dataset);
            timer.Stop();
            long t2 = timer.ElapsedMilliseconds;
            Console.WriteLine($"GPU: {t1}, CPU: {t2}");

            // Checks
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }
    }
}
