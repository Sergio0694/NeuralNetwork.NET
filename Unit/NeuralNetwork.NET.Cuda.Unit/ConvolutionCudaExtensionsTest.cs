using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Convolution;
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
            double[,] source = new double[1, square];
            Buffer.BlockCopy(m, 0, source, 0, sizeof(double) * square);
            double[][,] m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k);
            double[,] unpacked = m_gpu.MergeRows();
            Assert.IsTrue(unpacked.ContentEquals(r));
        }

        [TestMethod]
        public void Convolute3x3_2()
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
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k).MergeRows(),
                unpacked = new double[inner, inner];
            Buffer.BlockCopy(m_gpu, 0, unpacked, 0, sizeof(double) * innerSquare);
            Assert.IsTrue(unpacked.ContentEquals(r));
        }

        [TestMethod]
        public void Convolute3x3_3()
        {
            Random r = new Random();
            int 
                size = 3, 
                area = size * size, 
                bytesize = sizeof(double) * area,
                inner = size - 2,
                innerBytesize = sizeof(double) * inner;
            double[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new double[1, area * 2],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new double[inner, inner],
                unpacked2 = new double[inner, inner];
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
                bytesize = sizeof(double) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(double) * innerArea;
            double[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new double[2, area],
                m1 = r.NextXavierMatrix(size, size),
                m2 = r.NextXavierMatrix(size, size),
                check1 = ConvolutionExtensions.Convolute3x3(m1, k),
                check2 = ConvolutionExtensions.Convolute3x3(m2, k);
            Buffer.BlockCopy(m1, 0, source, 0, bytesize);
            Buffer.BlockCopy(m2, 0, source, bytesize, bytesize);
            double[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 1, k).MergeRows(),
                unpacked1 = new double[inner, inner],
                unpacked2 = new double[inner, inner];
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
                bytesize = sizeof(double) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(double) * innerArea;
            double[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new double[2, area * 2],
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
            double[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new double[inner, inner],
                unpacked2 = new double[inner, inner],
                unpacked3 = new double[inner, inner],
                unpacked4 = new double[inner, inner];
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
                bytesize = sizeof(double) * area,
                inner = size - 2,
                innerArea = inner * inner,
                innerBytesize = sizeof(double) * innerArea;
            double[,]
                k =
                {
                    { 1, -1, -1 },
                    { -1, 1, -1 },
                    { -1, -1, 1 }
                },
                source = new double[2, area * 2],
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
            double[,]
                m_gpu = ConvolutionGpuExtensions.Convolute3x3(source, 2, k).MergeRows(),
                unpacked1 = new double[inner, inner],
                unpacked2 = new double[inner, inner],
                unpacked3 = new double[inner, inner],
                unpacked4 = new double[inner, inner];
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
            double[,] dataset = r.NextXavierMatrix(1, 9);
            IReadOnlyList<double[,]> volume = dataset.Extract3DVolume();
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                v => v.Expand(ConvolutionExtensions.Convolute3x3,
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss));
            ConvolutionsStack[] processed = pipeline.Process(volume).ToArray();
            double[,] result_cpu = ConvolutionPipeline.ConvertToMatrix(processed);
            double[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
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
            double[,] dataset = r.NextXavierMatrix(1, 9);
            IReadOnlyList<double[,]> volume = dataset.Extract3DVolume();
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                v => v.Expand(ConvolutionExtensions.Convolute3x3,
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                v => v.Process(ConvolutionExtensions.ReLU));
            ConvolutionsStack[] processed = pipeline.Process(volume).ToArray();
            double[,] result_cpu = ConvolutionPipeline.ConvertToMatrix(processed);
            double[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
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
            double[,] dataset = r.NextXavierMatrix(2, 9);
            IReadOnlyList<double[,]> volume = dataset.Extract3DVolume();
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 3*3*1 pixels >> 1*1*3
                v => v.Expand(ConvolutionExtensions.Convolute3x3,
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                v => v.Process(ConvolutionExtensions.ReLU));
            ConvolutionsStack[] processed = pipeline.Process(volume).ToArray();
            double[,] result_cpu = ConvolutionPipeline.ConvertToMatrix(processed);
            double[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
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
            double[,] dataset = r.NextXavierMatrix(10, 81);
            IReadOnlyList<double[,]> volume = dataset.Extract3DVolume();
            ConvolutionPipeline pipeline = new ConvolutionPipeline(

                // 3 kernels, 9*9*1 pixels >> 7*7*3
                v => v.Expand(ConvolutionExtensions.Convolute3x3,
                    KernelsCollection.TopSobel,
                    KernelsCollection.BottomSobel,
                    KernelsCollection.TopLeftEmboss),
                v => v.Process(ConvolutionExtensions.ReLU),
                v => v.Process(ConvolutionExtensions.Pool2x2)); // 7*7*3 >> 4*4*3
            ConvolutionsStack[] processed = pipeline.Process(volume).ToArray();
            double[,] result_cpu = ConvolutionPipeline.ConvertToMatrix(processed);
            double[,] p1 = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.TopLeftEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(p1);
            double[,] result_gpu = ConvolutionGpuExtensions.Pool2x2(p1, 3);
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }

        [TestMethod]
        public void PipelineTest5()
        {
            // CPU pipeline
            ConvolutionPipeline pipeline = new ConvolutionPipeline(
            v => v.Expand(ConvolutionExtensions.Convolute3x3, // 10 kernels, 28*28*1 pixels >> 26*26*10
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomLeftEmboss,
                KernelsCollection.TopRightEmboss,
                KernelsCollection.TopLeftEmboss,
                KernelsCollection.BottomRightEmboss),
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2), // 26*26*10 >> 13*13*10,
            v => v.Expand(ConvolutionExtensions.Convolute3x3,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen),// 13*13*10 >> 11*11*60
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2), // 11*11*60 >> 6*6*60,
            v => v.Expand(ConvolutionExtensions.Convolute3x3,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomRightEmboss,
                KernelsCollection.TopLeftEmboss), // 6*6*60 >> 4*4*480
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2), // 4*4*480 >> 2*2*480
            v => v.Process(ConvolutionExtensions.Pool2x2)); // 2*2*480 >> 1*1*480

            // Setup and CPU calculation
            Random r = new Random();
            double[,] dataset = r.NextXavierMatrix(1000, 784);
            IReadOnlyList<double[,]> volume = dataset.Extract3DVolume();
            ConvolutionsStack[] processed = pipeline.Process(volume).ToArray();
            double[,] result_cpu = ConvolutionPipeline.ConvertToMatrix(processed);

            // GPU
            double[,] result_gpu = ConvolutionGpuExtensions.Convolute3x3(dataset, 1,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomLeftEmboss,
                KernelsCollection.TopRightEmboss,
                KernelsCollection.TopLeftEmboss,
                KernelsCollection.BottomRightEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(result_gpu);
            result_gpu = ConvolutionGpuExtensions.Pool2x2(result_gpu, 10);
            result_gpu = ConvolutionGpuExtensions.Convolute3x3(result_gpu, 10,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen).MergeRows();
            ConvolutionGpuExtensions.ReLU(result_gpu);
            result_gpu = ConvolutionGpuExtensions.Pool2x2(result_gpu, 60);
            result_gpu = ConvolutionGpuExtensions.Convolute3x3(result_gpu, 60,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomRightEmboss,
                KernelsCollection.TopLeftEmboss).MergeRows();
            ConvolutionGpuExtensions.ReLU(result_gpu);
            result_gpu = ConvolutionGpuExtensions.Pool2x2(result_gpu, 480);
            result_gpu = ConvolutionGpuExtensions.Pool2x2(result_gpu, 480);

            // Checks
            Assert.IsTrue(result_cpu.GetLength(0) == result_gpu.GetLength(0));
            Assert.IsTrue(result_cpu.GetLength(1) == result_gpu.GetLength(1));
            Assert.IsTrue(result_cpu.ContentEquals(result_gpu));
        }
    }
}
