using Alea;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN GPU extension methods
    /// </summary>
    [TestClass]
    [TestCategory(nameof(GpuExtensionsTest))]
    public class GpuExtensionsTest
    {
        [TestMethod]
        public void CudaSupport()
        {
            Assert.IsTrue(CuDnnNetworkLayers.IsCudaSupportAvailable);
        }

        [TestMethod]
        public void CopyToRows()
        {
            float[] test = {1,2,3,4,5,6,7,8,9};
            Tensor.NewZeroed(3, 10, out Tensor tensor);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory<float> m_gpu = gpu.AllocateDevice(test))
            {
                m_gpu.CopyTo(tensor, 5, 3);
            }
            float[,] expected =
            {
                { 0, 0, 0, 0, 0, 1, 2, 3, 0, 0 },
                { 0, 0, 0, 0, 0, 4, 5, 6, 0, 0 },
                { 0, 0, 0, 0, 0, 7, 8, 9, 0, 0 }
            };
            Assert.IsTrue(tensor.ToArray2D().ContentEquals(expected));
        }

        [TestMethod]
        public void AllocateDeviceRows()
        {
            float[,] source =
            {
                { 0, 0, 0, 0, 0, 1, 2, 3, 0, 0 },
                { 0, 0, 0, 0, 0, 4, 5, 6, 0, 0 },
                { 0, 0, 0, 0, 0, 7, 8, 9, 0, 0 }
            };
            Tensor.From(source, out Tensor tensor);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory<float> m_gpu = gpu.AllocateDevice(tensor, 5, 3))
            {
                float[]
                    copy = Gpu.CopyToHost(m_gpu),
                    expected = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
                Assert.IsTrue(copy.ContentEquals(expected));
            }
        }
    }
}
