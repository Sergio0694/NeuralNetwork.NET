using System;
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

        private static unsafe void TestForward(NetworkLayerBase cpu, NetworkLayerBase gpu, float[,] x)
        {
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
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
                Tensor.Reshape(pd_1, delta_1.GetLength(0), delta_1.GetLength(1), out Tensor delta_1t);
                Tensor.Reshape(pz, z.GetLength(0), z.GetLength(1), out Tensor zt);
                zt.Duplicate(out Tensor zt2);
                cpu.Backpropagate(Tensor.Null, delta_1t, zt, ActivationFunctions.LeCunTanhPrime);
                gpu.Backpropagate(Tensor.Null, delta_1t, zt2, ActivationFunctions.LeCunTanhPrime);
                Assert.IsTrue(zt.ContentEquals(zt2));
            }
        }

        private static unsafe void TestGradient(WeightedLayerBase cpu, WeightedLayerBase gpu, float[,] x, float[,] delta)
        {
            fixed (float* px = x, pdelta = delta)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
                Tensor.Reshape(pdelta, delta.GetLength(0), delta.GetLength(1), out Tensor deltat);
                cpu.ComputeGradient(xt, deltat, out Tensor dJdw_cpu, out Tensor dJdb_cpu);
                gpu.ComputeGradient(xt, deltat, out Tensor dJdw_gpu, out Tensor dJdb_gpu);
                Assert.IsTrue(dJdw_cpu.ContentEquals(dJdw_gpu));
                Assert.IsTrue(dJdb_cpu.ContentEquals(dJdb_gpu, 1e-4f, 1e-5f)); // The cuDNN ConvolutionBackwardBias is not always as precise as the CPU version
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void FullyConnectedBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 127),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void FullyConnectedGradient()
        {
            float[,]
                x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250),
                delta = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 127);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestGradient(cpu, gpu, x, delta);
        }

        #endregion

        #region Softmax

        [TestMethod]
        public void SoftmaxForward()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void SoftmaxBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 127),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void SoftmaxGradient()
        {
            float[,]
                a = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250),
                delta = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 127);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestGradient(cpu, gpu, a, delta);
        }

        [TestMethod]
        public unsafe void SoftmaxBackwardOutput()
        {
            float[,]
                x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 250),
                y = new float[400, 127];
            for (int i = 0; i < 400; i++)
                y[i, ThreadSafeRandom.NextInt(max: 127)] = 1;
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            fixed (float* px = x, py = y)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xt);
                cpu.Forward(xt, out Tensor z, out Tensor a);
                a.Duplicate(out Tensor a2);
                Tensor.Reshape(py, y.GetLength(0), y.GetLength(1), out Tensor yt);
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(127, 58 * 58 * 3);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeakyReLU, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public unsafe void ConvolutionBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 54 * 54 * 20, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(127, 54 * 54 * 20),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(127, 58 * 58 * 3);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh);
            fixed (float* pz = z)
            {
                Tensor.Reshape(pz, z.GetLength(0), z.GetLength(1), out Tensor zTensor);
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
                x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(127, 58 * 58 * 3),
                delta = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 54 * 54 * 5, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(127, 54 * 54 * 5);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 5, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh);
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(400, 58 * 58 * 3);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), PoolingInfo.Default, ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, PoolingInfo.Default, ActivationFunctionType.LeakyReLU);
            TestForward(cpu, gpu, x);
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
            cpu.Backpropagate(x, delta, x1, ActivationFunctions.LeakyReLUPrime);
            gpu.Backpropagate(x, delta, x2, ActivationFunctions.LeakyReLUPrime);
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
            x.Free();
            x1.Free();
            x2.Free();
            z.Free();
            delta.Free();
        }

        #endregion
    }
}
