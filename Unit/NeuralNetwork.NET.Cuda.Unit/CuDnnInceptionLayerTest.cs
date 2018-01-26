using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using System;
using System.Runtime.CompilerServices;
using NeuralNetworkNET.Networks.Layers.Cuda;
using NeuralNetworkNET.Networks.Layers.Initialization;
using SixLabors.ImageSharp.PixelFormats;

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
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(10), 32 * 32 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(10, 32 * 32 * 3);
            CuDnnConvolutionalLayer conv = new CuDnnConvolutionalLayer(TensorInfo.Image<Rgb24>(32, 32), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian);
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
                Tensor.Like(xTensor, out Tensor dx1);
                Tensor.Like(xTensor, out Tensor dx2);
                conv.Backpropagate(xTensor, zConv, aConv, dx1, out Tensor dJdw1, out Tensor dJdb1);
                inception.Backpropagate(xTensor, zInc, aInc, dx2, out Tensor dJdw2, out Tensor dJdb2);
                Assert.IsTrue(dx1.ContentEquals(dx2));
                Tensor.Reshape((float*)dJdw2.Ptr.ToPointer(), 1, dJdw1.Size, out dJdw2);
                Tensor.Reshape((float*)dJdb2.Ptr.ToPointer(), 1, dJdb1.Size, out dJdb2);
                Assert.IsTrue(dJdw1.ContentEquals(dJdw2, 1e-5f));
                Assert.IsTrue(dJdb1.ContentEquals(dJdb2, 1e-5f));

                // Cleanup
                Tensor.Free(zConv, aConv, zInc, aInc, reshaped, dx1, dx2, dJdw1, dJdw2, dJdb1, dJdb2);
            }
        }

        [TestMethod]
        public unsafe void Inception3x3Pipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(10), 32 * 32 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(10, 32 * 32 * 3);
            CuDnnConvolutionalLayer
                conv1 = new CuDnnConvolutionalLayer(TensorInfo.Image<Rgb24>(32, 32), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian),
                conv2 = new CuDnnConvolutionalLayer(conv1.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 1, 1), (3, 3), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.Image<Rgb24>(32, 32), InceptionInfo.New(10, 10, 10, 10, 10, PoolingMode.Max, 10));
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
                Tensor.Like(aTemp, out Tensor conv2dx);
                Tensor.Like(xTensor, out Tensor conv1dx);
                Tensor.Like(xTensor, out Tensor incdx);
                conv2.Backpropagate(aTemp, zConv, aConv, conv2dx, out Tensor conv2dJdw, out Tensor conv2dJdb);
                conv1.Backpropagate(xTensor, zTemp, conv2dx, conv1dx, out Tensor conv1dJdw, out Tensor conv1dJdb);
                inception.Backpropagate(xTensor, zInc, aInc, incdx, out Tensor incDjdw, out Tensor incdJdb);
                Assert.IsTrue(incdx.ContentEquals(conv1dx));

                // Gradient
                Tensor.Reshape((float*)incDjdw.Ptr.ToPointer() + 30, 1, conv1dJdw.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)incdJdb.Ptr.ToPointer() + 10, 1, conv1dJdb.Size, out Tensor dJdbInc0);
                Assert.IsTrue(conv1dJdw.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(conv1dJdb.ContentEquals(dJdbInc0, 1e-5f));
                Tensor.Reshape((float*)incDjdw.Ptr.ToPointer() + 30 + conv1dJdw.Size, 1, conv2dJdw.Size, out Tensor dJdwInc1);
                Tensor.Reshape((float*)incdJdb.Ptr.ToPointer() + 20, 1, conv2dJdb.Size, out Tensor dJdbInc1);
                Assert.IsTrue(conv2dJdw.ContentEquals(dJdwInc1, 1e-5f));
                Assert.IsTrue(conv2dJdb.ContentEquals(dJdbInc1, 1e-5f));

                // Cleanup
                Tensor.Free(zTemp, aTemp, zConv, aConv, zInc, aInc, reshaped, conv2dx, conv1dx, incdx, conv2dJdw, conv2dJdb, conv1dJdw, conv1dJdb, incDjdw, incdJdb);
            }
        }

        [TestMethod]
        public unsafe void Inception5x5Pipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(10), 12 * 12 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(10, 12 * 12 * 3);
            CuDnnConvolutionalLayer
                conv1 = new CuDnnConvolutionalLayer(TensorInfo.Image<Rgb24>(12, 12), ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian),
                conv2 = new CuDnnConvolutionalLayer(conv1.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation, 2, 2), (5, 5), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.Image<Rgb24>(12, 12), InceptionInfo.New(3, 2, 2, 10, 10, PoolingMode.Max, 2));
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
                Tensor.Like(aTemp, out Tensor conv2dx);
                Tensor.Like(xTensor, out Tensor conv1dx);
                Tensor.Like(xTensor, out Tensor incdx);
                conv2.Backpropagate(aTemp, zConv, aConv, conv2dx, out Tensor conv2dJdw, out Tensor conv2dJdb);
                conv1.Backpropagate(xTensor, zTemp, conv2dx, conv1dx, out Tensor conv1dJdw, out Tensor conv1dJdb);
                inception.Backpropagate(xTensor, zInc, aInc, incdx, out Tensor incDjdw, out Tensor incdJdb);
                Assert.IsTrue(incdx.ContentEquals(conv1dx));

                // Gradient
                Tensor.Reshape((float*)incDjdw.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2), 1, conv1dJdw.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)incdJdb.Ptr.ToPointer() + 7, 1, conv1dJdb.Size, out Tensor dJdbInc0);
                Assert.IsTrue(conv1dJdw.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(conv1dJdb.ContentEquals(dJdbInc0, 1e-5f));
                Tensor.Reshape((float*)incDjdw.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2) + conv1dJdw.Size, 1, conv2dJdw.Size, out Tensor dJdwInc1);
                Tensor.Reshape((float*)incdJdb.Ptr.ToPointer() + 17, 1, conv2dJdb.Size, out Tensor dJdbInc1);
                Assert.IsTrue(conv2dJdw.ContentEquals(dJdwInc1, 1e-5f));
                Assert.IsTrue(conv2dJdb.ContentEquals(dJdbInc1, 1e-5f));

                // Cleanup
                Tensor.Free(zTemp, aTemp, zConv, aConv, zInc, aInc, reshaped, conv2dx, conv1dx, incdx, conv2dJdw, conv2dJdb, conv1dJdw, conv1dJdb, incDjdw, incdJdb);
            }
        }

        [TestMethod]
        public unsafe void InceptionPoolPipeline()
        {
            float[,] x = WeightsProvider.NewFullyConnectedWeights(TensorInfo.Linear(10), 12 * 12 * 3, WeightsInitializationMode.GlorotNormal).AsSpan().AsMatrix(10, 12 * 12 * 3);
            CuDnnPoolingLayer pool = new CuDnnPoolingLayer(TensorInfo.Image<Rgb24>(12, 12), PoolingInfo.New(PoolingMode.Max, 3, 3, 1, 1, 1, 1), ActivationType.ReLU);
            CuDnnConvolutionalLayer conv = new CuDnnConvolutionalLayer(pool.OutputInfo, ConvolutionInfo.New(ConvolutionMode.CrossCorrelation), (1, 1), 10, ActivationType.ReLU, BiasInitializationMode.Gaussian);
            CuDnnInceptionLayer inception = new CuDnnInceptionLayer(TensorInfo.Image<Rgb24>(12, 12), InceptionInfo.New(3, 2, 2, 2, 2, PoolingMode.Max, 10));
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
                Tensor.Like(aTemp, out Tensor convdx);
                Tensor.Like(xTensor, out Tensor pooldx);
                Tensor.Like(xTensor, out Tensor incdx);
                conv.Backpropagate(aTemp, zConv, aConv, convdx, out Tensor convdJdw, out Tensor convdJdb);
                pool.Backpropagate(xTensor, zTemp, convdx, pooldx);
                inception.Backpropagate(xTensor, zInc, aInc, incdx, out Tensor incdJdw, out Tensor incdJdb);
                Assert.IsTrue(incdx.ContentEquals(pooldx));

                // Gradient
                Tensor.Reshape((float*)incdJdw.Ptr.ToPointer() + (3 * 3 + 3 * 2 + 3 * 3 * 2 * 2 + 3 * 2 + 5 * 5 * 2 * 2), 1, convdJdw.Size, out Tensor dJdwInc0);
                Tensor.Reshape((float*)incdJdb.Ptr.ToPointer() + 11, 1, convdJdb.Size, out Tensor dJdbInc0);
                Assert.IsTrue(convdJdw.ContentEquals(dJdwInc0, 1e-5f));
                Assert.IsTrue(convdJdb.ContentEquals(dJdbInc0, 1e-5f));

                // Cleanup
                Tensor.Free(zTemp, aTemp, zConv, aConv, zInc, aInc, reshaped, convdx, pooldx, incdx, convdJdw, convdJdb, incdJdw, incdJdb);
            }
        }
    }
}
