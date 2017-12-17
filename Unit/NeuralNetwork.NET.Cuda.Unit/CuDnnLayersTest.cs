using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN layers
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CuDnnLayersTest))]
    public class CuDnnLayersTest
    {
        [TestMethod]
        public unsafe void FullyConnectedForward()
        {
            float[,] x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(250, 127, ActivationFunctionType.LeCunTanh),
                gpu = new CuDnnFullyConnectedLayer(cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            fixed (float* px = x)
            {
                Tensor.Fix(px, 400, 250, out Tensor tensor);
                cpu.Forward(tensor, out Tensor z_cpu, out Tensor a_cpu);
                gpu.Forward(tensor, out Tensor z_gpu, out Tensor a_gpu);
                Assert.IsTrue(z_cpu.ContentEquals(z_gpu));
                Assert.IsTrue(a_cpu.ContentEquals(a_gpu));
            }
        }
    }
}
