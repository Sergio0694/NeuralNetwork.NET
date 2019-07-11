using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkDotNet.Core.Structs;
using NeuralNetworkDotNet.Cpu.cpuDNN;

namespace NeuralNetwork.NET.Cpu.Unit
{
    /// <summary>
    /// Tests for the <see cref="CpuBlas"/> <see langword="class"/>
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CpuBlasTests))]
    public class CpuBlasTests
    {
        /// <summary>
        /// Vector-matrix multiplication test
        /// </summary>
        [TestMethod]
        public void LinearMultiplication()
        {
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

            var tm = Tensor.From(m);
            var tv = Tensor.From(v);
            var y = Tensor.New(1, 4);
            CpuBlas.Multiply(tv, tm, y);
            Assert.IsTrue(Tensor.From(r).Equals(y));
        }

        /// <summary>
        /// Matrix-matrix multiplication test
        /// </summary>
        [TestMethod]
        public void SpatialMultiplication()
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

            var tm1 = Tensor.From(m1);
            var tm2 = Tensor.From(m2);
            var y = Tensor.New(2, 4);
            CpuBlas.Multiply(tm1, tm2, y);
            Assert.IsTrue(Tensor.From(r).Equals(y));
        }

        /// <summary>
        /// Element-wise matrix-matrix multiplication test
        /// </summary>
        [TestMethod]
        public void HadamardProductTest()
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

            var tm1 = Tensor.From(m1);
            var tm2 = Tensor.From(m2);
            var y = Tensor.Like(tm1);
            CpuBlas.MultiplyElementwise(tm1, tm2, y);
            Assert.IsTrue(Tensor.From(r).Equals(y));
        }

        /// <summary>
        /// Matrix transposition
        /// </summary>
        [TestMethod]
        public void Transposition()
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

            var tm = Tensor.From(m);
            var y = Tensor.New(4, 2);
            CpuBlas.Transpose(tm, y);
            Assert.IsTrue(Tensor.From(r).Equals(y));
        }
    }
}
