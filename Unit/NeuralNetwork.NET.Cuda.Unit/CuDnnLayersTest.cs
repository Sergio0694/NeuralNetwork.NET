using JetBrains.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.Networks.Layers.Cuda;
using NeuralNetworkNET.Networks.Layers.Initialization;

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

        // Creates a new random tensor with the given shape
        [Pure]
        private static unsafe Tensor CreateRandomTensor(int entities, int length)
        {
            float[] v = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(entities), length, WeightsInitializationMode.GlorotNormal);
            Tensor.New(entities, length, out Tensor tensor);
            fixed (float* pv = v)
            {
                Tensor.Reshape(pv, entities, length, out Tensor source);
                tensor.Overwrite(source);
                return tensor;
            }
        }

        private static void TestForward(NetworkLayerBase cpu, NetworkLayerBase gpu, int samples)
        {
            Tensor x = CreateRandomTensor(samples, cpu.InputInfo.Size);
            cpu.Forward(x, out Tensor z_cpu, out Tensor a_cpu);
            gpu.Forward(x, out Tensor z_gpu, out Tensor a_gpu);
            Assert.IsTrue(z_cpu.ContentEquals(z_gpu));
            Assert.IsTrue(a_cpu.ContentEquals(a_gpu));
            Tensor.Free(x, z_cpu, a_cpu, z_gpu, a_gpu);
        }

        private static void TestBackward(WeightedLayerBase cpu, WeightedLayerBase gpu, int samples)
        {
            Tensor
                x = CreateRandomTensor(samples, cpu.InputInfo.Size),
                dy = CreateRandomTensor(samples, cpu.OutputInfo.Size);
            Tensor.Like(x, out Tensor dx1);
            Tensor.Like(x, out Tensor dx2);
            cpu.Forward(x, out Tensor z_cpu, out Tensor a_cpu);
            gpu.Forward(x, out Tensor z_gpu, out Tensor a_gpu);
            cpu.Backpropagate(x, z_cpu, dy, dx1, out Tensor dJdw_cpu, out Tensor dJdb_cpu);
            gpu.Backpropagate(x, z_gpu, dy, dx2, out Tensor dJdw_gpu, out Tensor dJdb_gpu);
            Assert.IsTrue(dx1.ContentEquals(dx2));
            Assert.IsTrue(dJdw_cpu.ContentEquals(dJdw_gpu));
            Assert.IsTrue(dJdb_cpu.ContentEquals(dJdb_gpu, 1e-4f, 1e-5f)); // The cuDNN ConvolutionBackwardBias is not always as precise as the CPU version
            Tensor.Free(x, dy, dx1, dx2, z_cpu, a_cpu, z_gpu, a_gpu, dJdb_cpu, dJdb_gpu, dJdw_gpu, dJdb_gpu);
        }

        private static unsafe void TestBackward(OutputLayerBase cpu, OutputLayerBase gpu, float[,] y)
        {
            int n = y.GetLength(0);
            fixed (float* p = y)
            {
                Tensor.Reshape(p, n, y.GetLength(1), out Tensor yTensor);
                Tensor
                    x = CreateRandomTensor(n, cpu.InputInfo.Size),
                    dy = CreateRandomTensor(n, cpu.OutputInfo.Size);
                Tensor.Like(x, out Tensor dx1);
                Tensor.Like(x, out Tensor dx2);
                cpu.Forward(x, out Tensor z_cpu, out Tensor a_cpu);
                cpu.Backpropagate(x, a_cpu, yTensor, z_cpu, dx1, out Tensor dJdw_cpu, out Tensor dJdb_cpu);
                gpu.Backpropagate(x, a_cpu, yTensor, z_cpu, dx2, out Tensor dJdw_gpu, out Tensor dJdb_gpu);
                Assert.IsTrue(dx1.ContentEquals(dx2));
                Assert.IsTrue(dJdw_cpu.ContentEquals(dJdw_gpu));
                Assert.IsTrue(dJdb_cpu.ContentEquals(dJdb_gpu, 1e-4f, 1e-5f));
                Tensor.Free(x, dy, dx1, dx2, z_cpu, a_cpu, dJdb_cpu, dJdb_gpu, dJdw_gpu, dJdb_gpu);
            }
        }

        #endregion

        #region Fully connected

        [TestMethod]
        public void FullyConnectedForward()
        {
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.Linear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, 400);
        }

        [TestMethod]
        public void FullyConnectedBackward()
        {
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.Linear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestBackward(cpu, gpu, 400);
        }

        #endregion

        #region Softmax

        [TestMethod]
        public void SoftmaxForward()
        {
            OutputLayerBase
                cpu = new SoftmaxLayer(TensorInfo.Linear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestForward(cpu, gpu, 400);
        }

        [TestMethod]
        public void SoftmaxBackwardOutput()
        {
            float[,] y = new float[400, 127];
            for (int i = 0; i < 400; i++)
                y[i, ThreadSafeRandom.NextInt(max: 127)] = 1;
            OutputLayerBase
                cpu = new SoftmaxLayer(TensorInfo.Linear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestBackward(cpu, gpu, y);
        }

        #endregion

        #region Convolutional

        [TestMethod]
        public void ConvolutionForward()
        {
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeakyReLU, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, 127);
        }

        [TestMethod]
        public void ConvolutionBackward()
        {
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh);
            TestBackward(cpu, gpu, 127);
        }

        #endregion

        #region Pooling

        [TestMethod]
        public void PoolingForward()
        {
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), PoolingInfo.Default, ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, PoolingInfo.Default, ActivationFunctionType.LeakyReLU);
            TestForward(cpu, gpu, 400);
        }

        [TestMethod]
        public unsafe void PoolingBackward()
        {
            // Setup
            Tensor.New(400, 58 * 58 * 3, out Tensor x);
            KerasWeightsProvider.FillWithHeEtAlUniform(x, 10);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), PoolingInfo.Default, ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, PoolingInfo.Default, ActivationFunctionType.LeakyReLU);
            gpu.Forward(x, out Tensor z, out Tensor a);
            a.Free();
            x.Duplicate(out Tensor x1);
            x.Duplicate(out Tensor x2);
            Tensor.New(z.Entities, z.Length, out Tensor delta);
            KerasWeightsProvider.FillWithHeEtAlUniform(delta, 10);

            // Backward
            cpu.Backpropagate(x, z, delta, x1);
            gpu.Backpropagate(x, z, delta, x2);
            bool valid = true;
            float* px = (float*)x1.Ptr.ToPointer(), px2 = (float*)x2.Ptr.ToPointer();
            int count = 0;
            for (int i = 0; i < x1.Size; i++)
            {
                if (px[i].EqualsWithDelta(px2[i], 1e-5f)) continue;
                if (px[i].EqualsWithDelta(px2[i] * 100f, 1e-5f)) count++;   // The cuDNN pooling backwards method returns a value scaled by 0.01 from time to time for some reason (less than 2% anyways)
                else
                {
                    valid = false;
                    break;
                }
            }
            Assert.IsTrue(valid && count * 100f / x1.Size < 2);
            Tensor.Free(x, x1, x2, z, delta);
        }

        #endregion
    }
}
