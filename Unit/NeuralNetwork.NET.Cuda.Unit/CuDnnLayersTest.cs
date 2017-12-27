using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using System;
using System.Runtime.CompilerServices;

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
                cpu.Backpropagate(delta_1t, zt, ActivationFunctions.LeCunTanhPrime);
                gpu.Backpropagate(delta_1t, zt2, ActivationFunctions.LeCunTanhPrime);
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void FullyConnectedBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 127),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250);
            FullyConnectedLayer
                cpu = new FullyConnectedLayer(TensorInfo.CreateLinear(250), 127, ActivationFunctionType.LeCunTanh, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnFullyConnectedLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void FullyConnectedGradient()
        {
            float[,]
                x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250),
                delta = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 127);
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void SoftmaxBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 127),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestBackward(cpu, gpu, delta_1, z);
        }

        [TestMethod]
        public void SoftmaxGradient()
        {
            float[,]
                a = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250),
                delta = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 127, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 127);
            SoftmaxLayer
                cpu = new SoftmaxLayer(TensorInfo.CreateLinear(250), 127, WeightsInitializationMode.GlorotNormal, BiasInitializationMode.Gaussian),
                gpu = new CuDnnSoftmaxLayer(cpu.InputInfo, cpu.OutputInfo.Size, cpu.Weights, cpu.Biases);
            TestGradient(cpu, gpu, a, delta);
        }

        [TestMethod]
        public unsafe void SoftmaxBackwardOutput()
        {
            float[,]
                x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 250, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 250),
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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(127, 58 * 58 * 3);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeakyReLU, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, cpu.ActivationFunctionType);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public unsafe void ConvolutionBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 54 * 54 * 20, WeightsInitializationMode.GlorotNormal).AsMatrix(127, 54 * 54 * 20),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(127), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(127, 58 * 58 * 3);
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
        public void ConvolutionGradient()
        {
            // TODO: CPU gradient not implemented yet
            /* float[,]
                x = WeightsProvider.NewFullyConnectedWeights(127, 58 * 58 * 3),
                delta = WeightsProvider.NewFullyConnectedWeights(127, 54 * 54 * 20);
            ConvolutionalLayer
                cpu = new ConvolutionalLayer(new TensorInfo(58, 58, 3), ConvolutionInfo.Default, (5, 5), 20, ActivationFunctionType.LeCunTanh, BiasInitializationMode.Gaussian),
                gpu = new CuDnnConvolutionalLayer(cpu.InputInfo, ConvolutionInfo.Default, cpu.KernelInfo, cpu.OutputInfo, cpu.Weights, cpu.Biases, ActivationFunctionType.LeCunTanh);
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                gpu.Forward(xTensor, out Tensor z_gpu, out Tensor a_gpu);
                z_gpu.Free();
                a_gpu.Free();
            }
            TestGradient(cpu, gpu, x, delta); */
        }

        #endregion

        #region Pooling

        [TestMethod]
        public void PoolingForward()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 58 * 58 * 3);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), PoolingInfo.Default, ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, PoolingInfo.Default, ActivationFunctionType.LeakyReLU);
            TestForward(cpu, gpu, x);
        }

        [TestMethod]
        public void PoolingBackward()
        {
            float[,]
                delta_1 = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 29 * 29 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 29 * 29 * 3),
                z = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(400), 58 * 58 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(400, 58 * 58 * 3);
            PoolingLayer
                cpu = new PoolingLayer(new TensorInfo(58, 58, 3), PoolingInfo.Default, ActivationFunctionType.LeakyReLU),
                gpu = new CuDnnPoolingLayer(cpu.InputInfo, PoolingInfo.Default, ActivationFunctionType.LeakyReLU);
            TestBackward(cpu, gpu, delta_1, z);
        }

        #endregion

        #region Inception

        [TestMethod]
        public unsafe void InceptionForward1x1()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(10), 32 * 32 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(10, 32 * 32 * 3);
            CuDnnConvolutionalLayer conv = new CuDnnConvolutionalLayer(TensorInfo.CreateForRgbImage(32, 32), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(conv.InputInfo, InceptionInfo.New(10, 10, 10, 10, 10, PoolingMode.Max, 10));
            Buffer.BlockCopy(conv.Weights, 0, inception.Weights, 0, sizeof(float) * conv.Weights.Length);
            Buffer.BlockCopy(conv.Biases, 0, inception.Biases, 0, sizeof(float) * conv.Biases.Length);
            fixed (float* px = x)
            {
                // Forward + Z
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                conv.Forward(xTensor, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer(), preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));

                // A
                float* paInc = (float*)aInc.Ptr.ToPointer();
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));
                zConv.Free();
                aConv.Free();
                zInc.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void InceptionForward3x3Pipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(10), 32 * 32 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(10, 32 * 32 * 3);
            CuDnnConvolutionalLayer
                conv1 = new CuDnnConvolutionalLayer(TensorInfo.CreateForRgbImage(32, 32), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian),
                conv2 = new CuDnnConvolutionalLayer(conv1.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 1, 1), (3, 3), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.CreateForRgbImage(32, 32), InceptionInfo.New(10, 10, 10, 10, 10, PoolingMode.Max, 10));
            fixed (float* pw = inception.Weights)
                Unsafe.InitBlock(pw, 0, (uint)(sizeof(float) * inception.Weights.Length));
            Buffer.BlockCopy(conv1.Weights, 0, inception.Weights, sizeof(float) * 3 * 10, sizeof(float) * conv1.Weights.Length);
            Buffer.BlockCopy(conv2.Weights, 0, inception.Weights, sizeof(float) * 3 * 10 + sizeof(float) * conv1.Weights.Length, sizeof(float) * conv2.Weights.Length);
            Buffer.BlockCopy(conv1.Biases, 0, inception.Biases, sizeof(float) * 10, sizeof(float) * conv1.Biases.Length);
            Buffer.BlockCopy(conv2.Biases, 0, inception.Biases, sizeof(float) * 20, sizeof(float) * conv2.Biases.Length);
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                conv1.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                zTemp.Free();
                conv2.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 32 * 32 * 10, preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
                zConv.Free();
                zInc.Free();
                float* paInc = (float*)aInc.Ptr.ToPointer() + 32 * 32 * 10;
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void InceptionForward5x5Pipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(10), 12 * 12 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(10, 12 * 12 * 3);
            CuDnnConvolutionalLayer
                conv1 = new CuDnnConvolutionalLayer(TensorInfo.CreateForRgbImage(12, 12), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian),
                conv2 = new CuDnnConvolutionalLayer(conv1.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 2, 2), (5, 5), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.CreateForRgbImage(12, 12), InceptionInfo.New(3, 2, 2, 10, 10, PoolingMode.Max, 2));
            fixed (float* pw = inception.Weights)
                Unsafe.InitBlock(pw, 0, (uint)(sizeof(float) * inception.Weights.Length));
            Buffer.BlockCopy(conv1.Weights, 0, inception.Weights, sizeof(float) * (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2), sizeof(float) * conv1.Weights.Length);
            Buffer.BlockCopy(conv2.Weights, 0, inception.Weights, sizeof(float) * (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2 + conv1.Weights.Length), sizeof(float) * conv2.Weights.Length);
            Buffer.BlockCopy(conv1.Biases, 0, inception.Biases, sizeof(float) * (3 + 2 + 2), sizeof(float) * conv1.Biases.Length);
            Buffer.BlockCopy(conv2.Biases, 0, inception.Biases, sizeof(float) * (3 + 2 + 2 + 10), sizeof(float) * conv2.Biases.Length);
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                conv1.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                zTemp.Free();
                conv2.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 12 * 12 * (3 + 2), preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
                aTemp.Free();
                zConv.Free();
                zInc.Free();
                float* paInc = (float*)aInc.Ptr.ToPointer() + 12 * 12 * (3 + 2);
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void InceptionForwardPoolPipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(10), 12 * 12 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(10, 12 * 12 * 3);
            CuDnnPoolingLayer pool = new CuDnnPoolingLayer(TensorInfo.CreateForRgbImage(12, 12), PoolingInfo.New(PoolingMode.Max, 3, 3, 1, 1, 1, 1), ActivationFunctionType.ReLU);
            CuDnnConvolutionalLayer conv = new CuDnnConvolutionalLayer(pool.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.CreateForRgbImage(12, 12), InceptionInfo.New(3, 2, 2, 2, 2, PoolingMode.Max, 10));
            fixed (float* pw = inception.Weights)
                Unsafe.InitBlock(pw, 0, (uint)(sizeof(float) * inception.Weights.Length));
            Buffer.BlockCopy(conv.Weights, 0, inception.Weights, sizeof(float) * (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2 + 3 * 2 + 5 * 5 * 2 * 2), sizeof(float) * conv.Weights.Length);
            Buffer.BlockCopy(conv.Biases, 0, inception.Biases, sizeof(float) * (3 + 2 + 2 + 2 + 2), sizeof(float) * conv.Biases.Length);
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                pool.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                conv.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 12 * 12 * (3 + 2 + 2), preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
                zTemp.Free();
                aTemp.Free();
                zConv.Free();
                zInc.Free();
                float* paInc = (float*)aInc.Ptr.ToPointer() + 12 * 12 * (3 + 2 + 2);
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        #endregion
    }
}
