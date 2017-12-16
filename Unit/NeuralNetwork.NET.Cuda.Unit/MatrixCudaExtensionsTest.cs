using System.Diagnostics.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
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
        public unsafe void MultiplyWithSum()
        {
            float[,]
                m1 = ThreadSafeRandom.NextGlorotNormalMatrix(13, 5),
                m2 = ThreadSafeRandom.NextGlorotNormalMatrix(5, 4);
            float[] v = ThreadSafeRandom.NextGaussianVector(4);
            fixed (float* pm1 = m1)
            {
                Tensor.Fix(pm1, 13, 5, out Tensor m1Span);
                MatrixExtensions.MultiplyWithSum(m1Span, m2, v, out Tensor mul);
                Blas.MultiplyWithSum(m1Span, m2, v, out Tensor mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                mul.Free();
                mulGpu.Free();
            }
            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(1500, 800);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 40);
            v = ThreadSafeRandom.NextGaussianVector(40);
            fixed (float* pm1 = m1)
            {
                Tensor.Fix(pm1, 1500, 800, out Tensor m1Span);
                MatrixExtensions.MultiplyWithSum(m1Span, m2, v, out Tensor mul);
                Blas.MultiplyWithSum(m1Span, m2, v, out Tensor mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                mul.Free();
                mulGpu.Free();
            }
        }

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

        [TestMethod]
        public unsafe void Activation()
        {
            float[,] m = ThreadSafeRandom.NextGlorotNormalMatrix(20, 35);
            fixed (float* pm = m)
            {
                Tensor.Fix(pm, 20, 35, out Tensor mSpan);
                MatrixExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out Tensor activation);
                Blas.Activation(mSpan, ActivationFunctions.Sigmoid, out Tensor activationGpu);
                Assert.IsTrue(activation.ToArray2D().ContentEquals(activationGpu.ToArray2D(), 1e-4f));
                activation.Free();
                activationGpu.Free();
            }
            m = ThreadSafeRandom.NextGlorotNormalMatrix(1500, 800);
            fixed (float* pm = m)
            {
                Tensor.Fix(pm, 1500, 800, out Tensor mSpan);
                MatrixExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out Tensor activation);
                Blas.Activation(mSpan, ActivationFunctions.Sigmoid, out Tensor activationGpu);
                Assert.IsTrue(activation.ToArray2D().ContentEquals(activationGpu.ToArray2D(), 1e-4f));
                activation.Free();
                activationGpu.Free();
            }
        }

        [TestMethod]
        public unsafe void MultiplyAndInPlaceActivationPrimeAndHadamardProduct()
        {
            float[,]
                z = ThreadSafeRandom.NextGlorotNormalMatrix(40, 50),
                m1 = ThreadSafeRandom.NextGlorotNormalMatrix(40, 10),
                m2 = ThreadSafeRandom.NextGlorotNormalMatrix(10, 50),
                z2 = z.BlockCopy();
            fixed (float* pz = z, pm1 = m1, pm2 = m2, pz2 = z2)
            {
                Tensor.Fix(pz, 40, 50, out Tensor zSpan);
                Tensor.Fix(pm1, 40, 10, out Tensor m1Span);
                Tensor.Fix(pm2, 10, 50, out Tensor m2Span);
                Tensor.Fix(pz2, 40, 50, out Tensor z2Span);
                MatrixExtensions.InPlaceMultiplyAndHadamardProductWithActivationPrime(zSpan, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Blas.InPlaceMultiplyAndHadamardProductWithActivationPrime(z2Span, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Assert.IsTrue(zSpan.ToArray2D().ContentEquals(z2Span.ToArray2D(), 1e-4f));
            }
            z = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            z2 = z.BlockCopy();
            fixed (float* pz = z, pm1 = m1, pm2 = m2, pz2 = z2)
            {
                Tensor.Fix(pz, 200, 200, out Tensor zSpan);
                Tensor.Fix(pm1, 200, 200, out Tensor m1Span);
                Tensor.Fix(pm2, 200, 200, out Tensor m2Span);
                Tensor.Fix(pz2, 200, 200, out Tensor z2Span);
                MatrixExtensions.InPlaceMultiplyAndHadamardProductWithActivationPrime(zSpan, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Blas.InPlaceMultiplyAndHadamardProductWithActivationPrime(z2Span, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Assert.IsTrue(zSpan.ToArray2D().ContentEquals(z2Span.ToArray2D(), 1e-4f));
            }

        }
    }
}
