using System.Diagnostics.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the <see cref="MatrixCudaExtensionsTest"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(MatrixCudaExtensionsTest))]
    [SuppressMessage("ReSharper", "InvokeAsExtensionMethod")]
    public class MatrixCudaExtensionsTest
    {
        [TestMethod]
        public unsafe void TransposeAndMultiply()
        {
            float[,]
                m1 = ThreadSafeRandom.NextGlorotNormalMatrix(5, 13),
                m2 = ThreadSafeRandom.NextGlorotNormalMatrix(5, 4);
            fixed (float* pm1 = m1, pm2 = m2)
            {
                Tensor.Fix(pm1, 5, 13, out Tensor m1Span);
                Tensor.Fix(pm2, 5, 4, out Tensor m2Span);
                MatrixExtensions.Transpose(m1Span, out Tensor m1t);
                MatrixExtensions.Multiply(m1t, m2Span, out Tensor mul);
                Blas.TransposeAndMultiply(m1Span, m2Span, out Tensor mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                m1t.Free();
                mul.Free();
                mulGpu.Free();
            }

            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 1500);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 40);
            fixed (float* pm1 = m1, pm2 = m2)
            {
                Tensor.Fix(pm1, 800, 1500, out Tensor m1Span);
                Tensor.Fix(pm2, 800, 40, out Tensor m2Span);
                MatrixExtensions.Transpose(m1Span, out Tensor m1t);
                MatrixExtensions.Multiply(m1t, m2Span, out Tensor mul);
                Blas.TransposeAndMultiply(m1Span, m2Span, out Tensor mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                m1t.Free();
                mul.Free();
                mulGpu.Free();
            }
        }
    }
}
