using System.Diagnostics.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.Helpers;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Structs;

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
        public unsafe void ConvolutionForward1()
        {
            float[] biases = ThreadSafeRandom.NextGaussianVector(10);
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(100, 784),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(10, 25);
            fixed (float* psource = source)
            {
                FloatSpan2D.Fix(psource, 100, 784, out FloatSpan2D sourceSpan);
                ConvolutionExtensions.ConvoluteForward(sourceSpan, (28, 28, 1), kernels, (5, 5, 1), biases, out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteForward(sourceSpan, (28, 28, 1), kernels, (5, 5, 1), biases, out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }
        
        [TestMethod]
        public unsafe void ConvolutionForward2()
        {
            float[] biases = ThreadSafeRandom.NextGaussianVector(10);
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(100, 784 * 3),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(10, 25 * 3);
            fixed (float* psource = source)
            {
                FloatSpan2D.Fix(psource, 100, 784 * 3, out FloatSpan2D sourceSpan);
                ConvolutionExtensions.ConvoluteForward(sourceSpan, (28, 28, 3), kernels, (5, 5, 3), biases, out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteForward(sourceSpan, (28, 28, 3), kernels, (5, 5, 3), biases, out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionForward3()
        {
            float[] biases = ThreadSafeRandom.NextGaussianVector(10);
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(100, 1024 * 6),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(10, 9 * 6);
            fixed (float* psource = source)
            {
                FloatSpan2D.Fix(psource, 100, 1024 * 6, out FloatSpan2D sourceSpan);
                ConvolutionExtensions.ConvoluteForward(sourceSpan, (32, 32, 6), kernels, (3, 3, 6), biases, out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteForward(sourceSpan, (32, 32, 6), kernels, (3, 3, 6), biases, out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionBackwards1()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(30, 28 * 28),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(1, 24 * 24 * 6);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 30, 28 * 28, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 1, 24 * 24 * 6, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 1), kernelsSpan, (24, 24, 6), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 1), kernelsSpan, (24, 24, 6), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionBackwards2()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(25, 28 * 28 * 10),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(10, 24 * 24 * 3);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 25, 28 * 28 * 10, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 10, 24 * 24 * 3, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 10), kernelsSpan, (24, 24, 3), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 10), kernelsSpan, (24, 24, 3), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionBackwards3()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(20, 28 * 28 * 4),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(4, 24 * 24 * 8);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 20, 28 * 28 * 4, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 4, 24 * 24 * 8, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 4), kernelsSpan, (24, 24, 8), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteBackwards(sourceSpan, (28, 28, 4), kernelsSpan, (24, 24, 8), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionGradient1()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(30, 28 * 28 * 4),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(30, 24 * 24 * 7);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 30, 28 * 28 * 4, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 30, 24 * 24 * 7, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteGradient(sourceSpan, (28, 28, 4), kernelsSpan, (24, 24, 7), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteGradient(sourceSpan, (28, 28, 4), kernelsSpan, (24, 24, 7), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionGradient2()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(25, 28 * 28 * 10),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(25, 24 * 24 * 5);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 25, 28 * 28 * 10, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 25, 24 * 24 * 5, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteGradient(sourceSpan, (28, 28, 10), kernelsSpan, (24, 24, 5), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteGradient(sourceSpan, (28, 28, 10), kernelsSpan, (24, 24, 5), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void ConvolutionGradient3()
        {
            float[,]
                source = ThreadSafeRandom.NextGlorotNormalMatrix(10, 32 * 32 * 20),
                kernels = ThreadSafeRandom.NextGlorotNormalMatrix(10, 24 * 24 * 10);
            fixed (float* psource = source, pkernels = kernels)
            {
                FloatSpan2D.Fix(psource, 10, 32 * 32 * 20, out FloatSpan2D sourceSpan);
                FloatSpan2D.Fix(pkernels, 10, 24 * 24 * 10, out FloatSpan2D kernelsSpan);
                ConvolutionExtensions.ConvoluteGradient(sourceSpan, (32, 32, 20), kernelsSpan, (24, 24, 10), out FloatSpan2D result);
                ConvolutionGpuExtensions.ConvoluteGradient(sourceSpan, (32, 32, 20), kernelsSpan, (24, 24, 10), out FloatSpan2D resultGpu);
                Assert.IsTrue(result.ToArray2D().ContentEquals(resultGpu.ToArray2D(), 1e-4f));
                result.Free();
                resultGpu.Free();
            }
        }
    }
}
