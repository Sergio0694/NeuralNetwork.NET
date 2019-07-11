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
    }
}
