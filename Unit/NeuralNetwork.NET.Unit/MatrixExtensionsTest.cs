using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Unit
{
    /// <summary>
    /// Test class for the <see cref="MatrixExtensions"/> class
    /// </summary>
    [TestClass]
    [TestCategory(nameof(MatrixExtensionsTest))]
    public class MatrixExtensionsTest
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
                Tensor.Fix(pm, 4, 4, out Tensor mTensor);
                Tensor.Fix(pv, 1, 4, out Tensor vTensor);
                vTensor.Multiply(mTensor, out Tensor rTensor);
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
                Tensor.Fix(pm1, 2, 3, out Tensor m1Tensor);
                Tensor.Fix(pm2, 3, 4, out Tensor m2Tensor);
                m1Tensor.Multiply(m2Tensor, out Tensor result);
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
                Tensor.Fix(pm1, 3, 3, out Tensor m1Tensor);
                Tensor.Fix(pm2, 3, 3, out Tensor m2Tensor);
                m1Tensor.InPlaceHadamardProduct(m2Tensor);
                Assert.IsTrue(m1Tensor.ToArray2D().ContentEquals(r));
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
                Tensor.Fix(pm, 2, 4, out Tensor mTensor);
                mTensor.Transpose(out Tensor result);
                Assert.IsTrue(result.ToArray2D().ContentEquals(r));
                result.Free();
            }
        }

        /// <summary>
        /// Matrix array flattening
        /// </summary>
        [TestMethod]
        public void Flattening()
        {
            // Test values
            float[][,] mv =
            {
                new[,]
                {
                    { 1.0f, 2.0f },
                    { 3.0f, 4.0f }
                },
                new[,]
                {
                    { 0.1f, 0.2f },
                    { 0.3f, 0.4f }
                },
                new[,]
                {
                    { -1.0f, -2.0f },
                    { -3.0f, -4.0f }
                }
            };
            float[]
                r = { 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, -1.0f, -2.0f, -3.0f, -4.0f },
                t = mv.Flatten();
            Assert.IsTrue(t.ContentEquals(r));
        }

        [TestMethod]
        public void IndexOfMax1()
        {
            float[]
                v1 = new float[0],
                v2 = new float[1];
            Assert.IsTrue(v1.Argmax() == 0);
            Assert.IsTrue(v2.Argmax() == 0);
        }

        [TestMethod]
        public void IndexOfMax2()
        {
            float[]
                v1 = { 1f, 2f, 3f, 4f, 5f },
                v2 = { 99f, 11f },
                v3 = { -2f, -2.1f },
                v4 = { 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f };
            Assert.IsTrue(v1.Argmax() == 4);
            Assert.IsTrue(v2.Argmax() == 0);
            Assert.IsTrue(v3.Argmax() == 0);
            Assert.IsTrue(v4.Argmax() == 4);
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
