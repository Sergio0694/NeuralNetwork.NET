using Alea.cuDNN;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN layers
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CuDnnLayersTest))]
    public class CuDnnLayersTest
    {
        #region Helpers

        private static unsafe void TestForward(NetworkLayerBase cpu, NetworkLayerBase gpu, float[,] x)
        {
            fixed (float* px = x)
            {
                Tensor.Fix(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
                cpu.Forward(xt, out Tensor z_cpu, out Tensor a_cpu);
                gpu.Forward(xt, out Tensor z_gpu, out Tensor a_gpu);
                Assert.IsTrue(z_cpu.ContentEquals(z_gpu));
                Assert.IsTrue(a_cpu.ContentEquals(a_gpu));
                z_cpu.Free();
                a_cpu.Free();
                z_gpu.Free();
                a_gpu.Free();
            }
        }

        private static unsafe void TestBackward(NetworkLayerBase cpu, NetworkLayerBase gpu, float[,] delta_1, float[,] z)
        {
            fixed (float* pd_1 = delta_1, pz = z)
            {
                Tensor.Fix(pd_1, delta_1.GetLength(0), delta_1.GetLength(1), out Tensor delta_1t);
                Tensor.Fix(pz, z.GetLength(0), z.GetLength(1), out Tensor zt);
                zt.Duplicate(out Tensor zt2);
                cpu.Backpropagate(delta_1t, zt, ActivationFunctions.LeCunTanhPrime);
                gpu.Backpropagate(delta_1t, zt2, ActivationFunctions.LeCunTanhPrime);
                Assert.IsTrue(zt.ContentEquals(zt2));
            }
        }

        private static unsafe void TestGradient(WeightedLayerBase cpu, WeightedLayerBase gpu, float[,] x, float[,] delta)
        {
            fixed (float* px = x, pdelta = delta)
            {
                Tensor.Fix(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
                Tensor.Fix(pdelta, delta.GetLength(0), delta.GetLength(1), out Tensor deltat);
                cpu.ComputeGradient(xt, deltat, out Tensor dJdw_cpu, out Tensor dJdb_cpu);
                gpu.ComputeGradient(xt, deltat, out Tensor dJdw_gpu, out Tensor dJdb_gpu);
                Assert.IsTrue(dJdw_cpu.ContentEquals(dJdw_gpu));
                Assert.IsTrue(dJdb_cpu.ContentEquals(dJdb_gpu));
                dJdw_cpu.Free();
                dJdw_gpu.Free();
                dJdb_cpu.Free();
                dJdb_gpu.Free();
            }
        }

        #endregion

        #region Fully connected

        [TestMethod]
        public void FullyConnectedForward()
        {
            float[,] x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void FullyConnectedBackward()
        {
            float[,]
                delta_1 = ThreadSafeRandom.NextGlorotNormalMatrix(400, 127),
                z = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void FullyConnectedGradient()
        {
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250),
                delta = ThreadSafeRandom.NextGlorotNormalMatrix(400, 127);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestGradient(cpu, gpu, x, delta);
        }

        #endregion

        #region Softmax

        [TestMethod]
        public void SoftmaxForward()
        {
            float[,] x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.Weights, cpu.Biases);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void SoftmaxBackward()
        {
            float[,]
                delta_1 = ThreadSafeRandom.NextGlorotNormalMatrix(400, 127),
                z = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.Weights, cpu.Biases);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void SoftmaxGradient()
        {
            float[,]
                a = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250),
                delta = ThreadSafeRandom.NextGlorotNormalMatrix(400, 127);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.Weights, cpu.Biases);
            TestGradient(cpu, gpu, a, delta);
        }

        [TestMethod]
        public unsafe void SoftmaxBackwardOutput()
        {
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 250),
                y = new float[400, 127];
            for (int i = 0; i < 400; i++)
                y[i, ThreadSafeRandom.NextInt(max: 127)] = 1;
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.Weights, cpu.Biases);
            fixed (float* px = x, py = y)
            {
                Tensor.Fix(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
                cpu.Forward(xt, out Tensor z, out Tensor a);
                a.Duplicate(out Tensor a2);
                Tensor.Fix(py, y.GetLength(0), y.GetLength(1), out Tensor yt);
                cpu.Backpropagate(a, yt, z);
                gpu.Backpropagate(a2, yt, z);
                Assert.IsTrue(a.ContentEquals(a2));
                a.Free();
                a2.Free();
                z.Free();
            }
        }

        #endregion

        #region Convolutional

        [TestMethod]
        public void ConvolutionForward()
        {
            float[,] x = ThreadSafeRandom.NextGlorotNormalMatrix(127, 58 * 58 * 3);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), (5, 5), 20, ActivationFunctionType.LeakyReLU, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType, ConvolutionMode.CONVOLUTION);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public unsafe void ConvolutionBackward()
        {
            float[,]
                delta_1 = ThreadSafeRandom.NextGlorotNormalMatrix(127, 54 * 54 * 20),
                z = ThreadSafeRandom.NextGlorotNormalMatrix(127, 58 * 58 * 3);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), (5, 5), 20, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh, ConvolutionMode.CONVOLUTION);
            fixed (float* pz = z)
            {
                Tensor.Fix(pz, z.GetLength(0), z.GetLength(1), out Tensor zTensor);
                gpu.Forward(zTensor, out Tensor z_gpu, out Tensor a_gpu);
                z_gpu.Free();
                a_gpu.Free();
            }
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public unsafe void ConvolutionGradient()
        {
            float[,]
                x = ThreadSafeRandom.NextGlorotNormalMatrix(127, 58 * 58 * 3),
                delta = ThreadSafeRandom.NextGlorotNormalMatrix(127, 54 * 54 * 20);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), (5, 5), 20, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh, ConvolutionMode.CONVOLUTION);
            fixed (float* px = x)
            {
                Tensor.Fix(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                gpu.Forward(xTensor, out Tensor z_gpu, out Tensor a_gpu);
                z_gpu.Free();
                a_gpu.Free();
            }
            TestGradient(cpu, gpu, x, delta);
        }

        #endregion

        #region Pooling

        [TestMethod]
        public void PoolingForward()
        {
            float[,] x = ThreadSafeRandom.NextGlorotNormalMatrix(400, 58 * 58 * 3);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, ActivationFunctionType.LeakyReLU);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void PoolingBackward()
        {
            float[,]
                delta_1 = ThreadSafeRandom.NextGlorotNormalMatrix(400, 29 * 29 * 3),
                z = ThreadSafeRandom.NextGlorotNormalMatrix(400, 58 * 58 * 3);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, ActivationFunctionType.LeakyReLU);
            TestBackward(cpu, gpu, delta_1, z);
        }

        #endregion
    }
}
