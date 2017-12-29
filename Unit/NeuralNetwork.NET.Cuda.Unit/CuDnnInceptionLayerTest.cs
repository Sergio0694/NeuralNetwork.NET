using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.Cuda.Unit
{
    /// <summary>
    /// Test class for the cuDNN inception layer
    /// </summary>
    [TestClass]
    [TestCategory(nameof(CuDnnInceptionLayerTest))]
    public class CuDnnInceptionLayerTest
    {
        [TestMethod]
        public unsafe void Inception1x1()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.CreateLinear(10), 32 * 32 * 3, WeightsInitializationMode.GlorotNormal).AsMatrix(10, 32 * 32 * 3);
            CuDnnConvolutionalLayer conv = new CuDnnConvolutionalLayer(TensorInfo.CreateForRgbImage(32, 32), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationFunctionType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(conv.InputInfo, InceptionInfo.New(10, 10, 10, 10, 10, PoolingMode.Max, 10));
            fixed (float* pw = inception.Weights)
                Unsafe.InitBlock(pw, 0, (uint)(sizeof(float) * inception.Weights.Length));
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

                // Backpropagate
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor z1);
                KerasWeightsProvider.FillWithHeEtAlUniform(z1, 10);
                z1.Duplicate(out Tensor z2);
                conv.Backpropagate(aConv, z1, ActivationFunctions.ReLUPrime);
                inception.Backpropagate(aInc, z2, ActivationFunctions.ReLUPrime);
                Assert.IsTrue(z1.ContentEquals(z2));

                // Gradient
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor a);
                KerasWeightsProvider.FillWithHeEtAlUniform(a, 10);
                conv.ComputeGradient(a, aConv, out Tensor dJdwConv, out Tensor dJdbConv);
                inception.ComputeGradient(a, aInc, out Tensor dJdwInc, out Tensor dJdbInc);
                Tensor.New(1, dJdwConv.Length, out Tensor dJdwInc0);
                Buffer.MemoryCopy((float*)dJdwInc.Ptr.ToPointer(), (float*)dJdwInc0.Ptr.ToPointer(), sizeof(float) * dJdwInc0.Size, sizeof(float) * dJdwInc0.Size);
                Tensor.New(1, dJdbConv.Length, out Tensor dJdbInc0);
                Buffer.MemoryCopy((float*)dJdbInc.Ptr.ToPointer(), (float*)dJdbInc0.Ptr.ToPointer(), sizeof(float) * dJdbInc0.Size, sizeof(float) * dJdbInc0.Size);
                Assert.IsTrue(dJdwConv.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(dJdbConv.ContentEquals(dJdbInc0, 1e-5f));

                // Cleanup
                dJdwConv.Free();
                dJdbConv.Free();
                dJdwInc.Free();
                dJdbInc.Free();
                dJdwInc0.Free();
                dJdbInc0.Free();
                z1.Free();
                z2.Free();
                zConv.Free();
                aConv.Free();
                zInc.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void Inception3x3Pipeline()
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
                // Forward + Z
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                conv1.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                conv2.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 32 * 32 * 10, preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
                
                // A
                float* paInc = (float*)aInc.Ptr.ToPointer() + 32 * 32 * 10;
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));

                // Backpropagation
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor z1);
                KerasWeightsProvider.FillWithHeEtAlUniform(z1, 10);
                z1.Duplicate(out Tensor z2);
                conv2.Backpropagate(aConv, zTemp, conv1.ActivationFunctions.ActivationPrime);
                conv1.Backpropagate(zTemp, z1, ActivationFunctions.ReLUPrime);
                inception.Backpropagate(aInc, z2, ActivationFunctions.ReLUPrime);
                Assert.IsTrue(z1.ContentEquals(z2));

                // Gradient
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor a);
                KerasWeightsProvider.FillWithHeEtAlUniform(a, 10);
                conv1.ComputeGradient(a, zTemp, out Tensor dJdwConv1, out Tensor dJdbConv1);
                conv2.ComputeGradient(aTemp, aConv, out Tensor dJdwConv2, out Tensor dJdbConv2);
                inception.ComputeGradient(a, aInc, out Tensor dJdwInc, out Tensor dJdbInc);
                Tensor.Reshape((float*)dJdwInc.Ptr.ToPointer() + 30, 1, dJdwConv1.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)dJdbInc.Ptr.ToPointer() + 10, 1, dJdbConv1.Size, out Tensor dJdbInc0);
                Assert.IsTrue(dJdwConv1.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(dJdbConv1.ContentEquals(dJdbInc0, 1e-5f));
                Tensor.Reshape((float*)dJdwInc.Ptr.ToPointer() + 30 + dJdwConv1.Size, 1, dJdwConv2.Size, out Tensor dJdwInc1);
                Tensor.Reshape((float*)dJdbInc.Ptr.ToPointer() + 20, 1, dJdbConv2.Size, out Tensor dJdbInc1);
                Assert.IsTrue(dJdwConv2.ContentEquals(dJdwInc1, 1e-5f));
                Assert.IsTrue(dJdbConv2.ContentEquals(dJdbInc1, 1e-5f));

                // Cleanup
                z1.Free();
                z2.Free();
                zTemp.Free();
                zConv.Free();
                zInc.Free();
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void Inception5x5Pipeline()
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
                // Forwaard + Z
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                conv1.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                conv2.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 12 * 12 * (3 + 2), preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
                
                // A
                float* paInc = (float*)aInc.Ptr.ToPointer() + 12 * 12 * (3 + 2);
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));

                // Backpropagation
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor z1);
                KerasWeightsProvider.FillWithHeEtAlUniform(z1, 10);
                z1.Duplicate(out Tensor z2);
                conv2.Backpropagate(aConv, zTemp, conv1.ActivationFunctions.ActivationPrime);
                conv1.Backpropagate(zTemp, z1, ActivationFunctions.ReLUPrime);
                inception.Backpropagate(aInc, z2, ActivationFunctions.ReLUPrime);
                Assert.IsTrue(z1.ContentEquals(z2));

                // Gradient
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor a);
                KerasWeightsProvider.FillWithHeEtAlUniform(a, 10);
                conv1.ComputeGradient(a, zTemp, out Tensor dJdwConv1, out Tensor dJdbConv1);
                conv2.ComputeGradient(aTemp, aConv, out Tensor dJdwConv2, out Tensor dJdbConv2);
                inception.ComputeGradient(a, aInc, out Tensor dJdwInc, out Tensor dJdbInc);
                Tensor.Reshape((float*)dJdwInc.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2), 1, dJdwConv1.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)dJdbInc.Ptr.ToPointer() + 7, 1, dJdbConv1.Size, out Tensor dJdbInc0);
                Assert.IsTrue(dJdwConv1.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(dJdbConv1.ContentEquals(dJdbInc0, 1e-5f));
                Tensor.Reshape((float*)dJdwInc.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2) + dJdwConv1.Size, 1, dJdwConv2.Size, out Tensor dJdwInc1);
                Tensor.Reshape((float*)dJdbInc.Ptr.ToPointer() + 17, 1, dJdbConv2.Size, out Tensor dJdbInc1);
                Assert.IsTrue(dJdwConv2.ContentEquals(dJdwInc1, 1e-5f));
                Assert.IsTrue(dJdbConv2.ContentEquals(dJdbInc1, 1e-5f));

                // Cleanup
                zTemp.Free();
                aTemp.Free();
                zConv.Free();
                zInc.Free();
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }

        [TestMethod]
        public unsafe void InceptionPoolPipeline()
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
                // Forward + Z
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                pool.Forward(xTensor, out Tensor zTemp, out Tensor aTemp);
                conv.Forward(aTemp, out Tensor zConv, out Tensor aConv);
                inception.Forward(xTensor, out Tensor zInc, out Tensor aInc);
                Tensor.New(zConv.Entities, zConv.Length, out Tensor reshaped);
                float* pzInc = (float*)zInc.Ptr.ToPointer() + 12 * 12 * (3 + 2 + 2), preshaped = (float*)reshaped.Ptr.ToPointer();
                for (int i = 0; i < zConv.Entities; i++)
                    Buffer.MemoryCopy(pzInc + i * zInc.Length, preshaped + i * zConv.Length, sizeof(float) * zConv.Length, sizeof(float) * zConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(zConv));
               
                // A
                float* paInc = (float*)aInc.Ptr.ToPointer() + 12 * 12 * (3 + 2 + 2);
                for (int i = 0; i < aConv.Entities; i++)
                    Buffer.MemoryCopy(paInc + i * aInc.Length, preshaped + i * aConv.Length, sizeof(float) * aConv.Length, sizeof(float) * aConv.Length);
                Assert.IsTrue(reshaped.ContentEquals(aConv));

                // Backpropagation
                Tensor.New(xTensor.Entities, xTensor.Length, out Tensor z1);
                KerasWeightsProvider.FillWithHeEtAlUniform(z1, 10);
                z1.Duplicate(out Tensor z2);
                conv.Backpropagate(aConv, zTemp, pool.ActivationFunctions.ActivationPrime);
                pool.Backpropagate(zTemp, z1, ActivationFunctions.ReLUPrime);
                inception.Backpropagate(aInc, z2, ActivationFunctions.ReLUPrime);
                Assert.IsTrue(z1.ContentEquals(z2));

                // Gradient
                conv.ComputeGradient(aTemp, aConv, out Tensor dJdwConv, out Tensor dJdbConv);
                inception.ComputeGradient(xTensor, aInc, out Tensor dJdwInc, out Tensor dJdbInc);
                Tensor.Reshape((float*)dJdwInc.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2 + 3 * 2 + 5 * 5 * 2 * 2), 1, dJdwConv.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)dJdbInc.Ptr.ToPointer() + 11, 1, dJdbConv.Size, out Tensor dJdbInc0);
                Assert.IsTrue(dJdwConv.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(dJdbConv.ContentEquals(dJdbInc0, 1e-5f));

                // Cleanup
                zTemp.Free();
                aTemp.Free();
                zConv.Free();
                zInc.Free();
                aConv.Free();
                aInc.Free();
                reshaped.Free();
            }
        }
    }
}
