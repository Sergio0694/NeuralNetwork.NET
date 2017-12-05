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
                FloatSpan2D.Fix(pm1, 13, 5, out FloatSpan2D m1Span);
                MatrixExtensions.MultiplyWithSum(m1Span, m2, v, out FloatSpan2D mul);
                MatrixGpuExtensions.MultiplyWithSum(m1Span, m2, v, out FloatSpan2D mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                mul.Free();
                mulGpu.Free();
            }
            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(1500, 800);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 40);
            v = ThreadSafeRandom.NextGaussianVector(40);
            fixed (float* pm1 = m1)
            {
                FloatSpan2D.Fix(pm1, 1500, 800, out FloatSpan2D m1Span);
                MatrixExtensions.MultiplyWithSum(m1Span, m2, v, out FloatSpan2D mul);
                MatrixGpuExtensions.MultiplyWithSum(m1Span, m2, v, out FloatSpan2D mulGpu);
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
                FloatSpan2D.Fix(pm1, 5, 13, out FloatSpan2D m1Span);
                FloatSpan2D.Fix(pm2, 5, 4, out FloatSpan2D m2Span);
                MatrixExtensions.Transpose(m1Span, out FloatSpan2D m1t);
                MatrixExtensions.Multiply(m1t, m2Span, out FloatSpan2D mul);
                MatrixGpuExtensions.TransposeAndMultiply(m1Span, m2Span, out FloatSpan2D mulGpu);
                Assert.IsTrue(mul.ToArray2D().ContentEquals(mulGpu.ToArray2D(), 1e-4f));
                m1t.Free();
                mul.Free();
                mulGpu.Free();
            }

            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 1500);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(800, 40);
            fixed (float* pm1 = m1, pm2 = m2)
            {
                FloatSpan2D.Fix(pm1, 800, 1500, out FloatSpan2D m1Span);
                FloatSpan2D.Fix(pm2, 800, 40, out FloatSpan2D m2Span);
                MatrixExtensions.Transpose(m1Span, out FloatSpan2D m1t);
                MatrixExtensions.Multiply(m1t, m2Span, out FloatSpan2D mul);
                MatrixGpuExtensions.TransposeAndMultiply(m1Span, m2Span, out FloatSpan2D mulGpu);
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
                FloatSpan2D.Fix(pm, 20, 35, out FloatSpan2D mSpan);
                MatrixExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out FloatSpan2D activation);
                MatrixGpuExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out FloatSpan2D activationGpu);
                Assert.IsTrue(activation.ToArray2D().ContentEquals(activationGpu.ToArray2D(), 1e-4f));
                activation.Free();
                activationGpu.Free();
            }
            m = ThreadSafeRandom.NextGlorotNormalMatrix(1500, 800);
            fixed (float* pm = m)
            {
                FloatSpan2D.Fix(pm, 1500, 800, out FloatSpan2D mSpan);
                MatrixExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out FloatSpan2D activation);
                MatrixGpuExtensions.Activation(mSpan, ActivationFunctions.Sigmoid, out FloatSpan2D activationGpu);
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
                FloatSpan2D.Fix(pz, 40, 50, out FloatSpan2D zSpan);
                FloatSpan2D.Fix(pm1, 40, 10, out FloatSpan2D m1Span);
                FloatSpan2D.Fix(pm2, 10, 50, out FloatSpan2D m2Span);
                FloatSpan2D.Fix(pz2, 40, 50, out FloatSpan2D z2Span);
                MatrixExtensions.InPlaceMultiplyAndHadamardProductWithActivationPrime(zSpan, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                MatrixGpuExtensions.MultiplyAndHadamardProductWithActivation(z2Span, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Assert.IsTrue(zSpan.ToArray2D().ContentEquals(z2Span.ToArray2D(), 1e-4f));
            }
            z = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            m1 = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            m2 = ThreadSafeRandom.NextGlorotNormalMatrix(200, 200);
            z2 = z.BlockCopy();
            fixed (float* pz = z, pm1 = m1, pm2 = m2, pz2 = z2)
            {
                FloatSpan2D.Fix(pz, 200, 200, out FloatSpan2D zSpan);
                FloatSpan2D.Fix(pm1, 200, 200, out FloatSpan2D m1Span);
                FloatSpan2D.Fix(pm2, 200, 200, out FloatSpan2D m2Span);
                FloatSpan2D.Fix(pz2, 200, 200, out FloatSpan2D z2Span);
                MatrixExtensions.InPlaceMultiplyAndHadamardProductWithActivationPrime(zSpan, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                MatrixGpuExtensions.MultiplyAndHadamardProductWithActivation(z2Span, m1Span, m2Span, ActivationFunctions.SigmoidPrime);
                Assert.IsTrue(zSpan.ToArray2D().ContentEquals(z2Span.ToArray2D(), 1e-4f));
            }

        }
    }
}
