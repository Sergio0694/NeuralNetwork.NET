using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="CpuBlas"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CpuBlasTest))]
    public class CpuBlasTest
    {
        /// <summary>
        /// Vector-matrix multiplication test
        /// </summary>
        [TestMethod]
        public unsafe void LinearMultiplication()
        {
            // Test values
            float[,] m =
            {
                { 1, 1, 1, 1 },
                { 0, 2, -1, 0 },
                { 1, 1, 1, 1 },
                { 0, 0, -1, 1 }
            };
            float[]
                v = { 1, 2, 0.1f, -2 },
                r = { 1.1f, 5.1f, 1.1f, -0.9f };
            fixed (float* pm = m, pv = v)
            {
                Tensor.Reshape(pm, 4, 4, out Tensor mTensor);
                Tensor.Reshape(pv, 1, 4, out Tensor vTensor);
                Tensor.New(1, 4, out Tensor rTensor);
                CpuBlas.Multiply(vTensor, mTensor, rTensor);
                Assert.IsTrue(rTensor.ToArray().ContentEquals(r));
                rTensor.Free();
            }
        }

        /// <summary>
        /// Matrix-matrix multiplication test
        /// </summary>
        [TestMethod]
        public unsafe void SpatialMultiplication()
        {
            // Test values
            float[,]
                m1 =
                {
                    { 1, 2, 3 },
                    { 5, 0.1f, -2 }
                },
                m2 =
                {
                    { 5, 2, -1, 3 },
                    { -5, 2, -7, 0.9f },
                    { 0.1f, 0.2f, -0.1f, 2 }
                },
                r =
                {
                    { -4.7f, 6.6f, -15.3f, 10.8f },
                    { 24.3f, 9.7999999999999989f, -5.5f, 11.09f }
                };
            fixed (float* pm1 = m1, pm2 = m2)
            {
                Tensor.Reshape(pm1, 2, 3, out Tensor m1Tensor);
                Tensor.Reshape(pm2, 3, 4, out Tensor m2Tensor);
                Tensor.New(2, 4, out Tensor result);
                CpuBlas.Multiply(m1Tensor, m2Tensor, result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        /// <summary>
        /// Element-wise matrix-matrix multiplication test
        /// </summary>
        [TestMethod]
        public unsafe void HadamardProductTest()
        {
            // Test values
            float[,]
                m1 =
                {
                    { 1, 2, 3 },
                    { 5, 1, -2 },
                    { 1, 2, 3 },
                },
                m2 =
                {
                    { 5, 2, -1 },
                    { -5, 2, -7 },
                    { 1, 2, 2 }
                },
                r =
                {
                    { 5, 4, -3 },
                    { -25, 2, 14 },
                    { 1, 4, 6 }
                };
            fixed (float* pm1 = m1, pm2 = m2)
            {
                Tensor.Reshape(pm1, 3, 3, out Tensor m1Tensor);
                Tensor.Reshape(pm2, 3, 3, out Tensor m2Tensor);
                Tensor.New(3, 3, out Tensor result);
                CpuBlas.MultiplyElementwise(m1Tensor, m2Tensor, result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        /// <summary>
        /// Matrix transposition
        /// </summary>
        [TestMethod]
        public unsafe void Transposition()
        {
            // Test values
            float[,]
                m =
                {
                    { 1, 1, 1, 1 },
                    { 0, 2, -1, 0 }
                },
                r =
                {
                    { 1, 0 },
                    { 1, 2 },
                    { 1, -1 },
                    { 1, 0 }
                };
            fixed (float* pm = m)
            {
                Tensor.Reshape(pm, 2, 4, out Tensor mTensor);
                Tensor.New(4, 2, out Tensor result);
                CpuBlas.Transpose(mTensor, result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        [TestMethod]
        public void IndexOfMax1()
        {
            Span<float> 
                v1 = new float[0],
                v2 = new float[1];
            Assert.IsTrue(v1.Argmax(float.MinValue) == 0);
            Assert.IsTrue(v2.Argmax(float.MinValue) == 0);
        }

        [TestMethod]
        public void IndexOfMax2()
        {
            Span<float>
                v1 = new[] { 1f, 2f, 3f, 4f, 5f },
                v2 = new[] { 99f, 11f },
                v3 = new[] { -2f, -2.1f },
                v4 = new[] { 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f };
            Assert.IsTrue(v1.Argmax(float.MinValue) == 4);
            Assert.IsTrue(v2.Argmax(float.MinValue) == 0);
            Assert.IsTrue(v3.Argmax(float.MinValue) == 0);
            Assert.IsTrue(v4.Argmax(float.MinValue) == 4);
        }

        [TestMethod]
        public void ToFormattedString()
        {
            float[,]
                empty = { { } },
                oneLine = { { 1.0f, 2.0f, 3.0f } },
                complete = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };
            String
                emptyString = "{ { } }",
                oneLineString = "{ { 1, 2, 3 } }",
                completeString = "{ { 1, 2, 3 },\n  { 4, 5, 6 } }";
            Assert.IsTrue(empty.ToFormattedString().Equals(emptyString));
            Assert.IsTrue(oneLine.ToFormattedString().Equals(oneLineString));
            Assert.IsTrue(complete.ToFormattedString().Equals(completeString));
        }
    }
}
