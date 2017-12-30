using System;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Cuda.Layers;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that exposes the available cuDNN network layer types
    /// </summary>
    public static class CuDnnNetworkLayers
    {
        /// <summary>
        /// Gets whether or not the Cuda acceleration is supported on the current system
        /// </summary>
        public static bool IsCudaSupportAvailable
        {
            [Pure]
            get
            {
                try
                {
                    // Calling this directly would could a crash in the <Module> loader due to the missing .dll files
                    return CuDnnSupportHelper.IsGpuAccelerationSupported();
                }
                catch (TypeInitializationException)
                {
                    // Missing .dll file
                    return false;
                }
            }
        }

        /// <summary>
        /// Creates a new fully connected layer with the specified number of input and output neurons, and the given activation function
        /// </summary>
        /// <param name="neurons">The number of output neurons</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory FullyConnected(
            int neurons, ActivationFunctionType activation,
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnFullyConnectedLayer(input, neurons, activation, weightsMode, biasMode);

        /// <summary>
        /// Creates a fully connected softmax output layer (used for classification problems with mutually-exclusive classes)
        /// </summary>
        /// <param name="outputs">The number of output neurons</param>
        /// <param name="weightsMode">The desired initialization mode for the weights in the network layer</param>
        /// <param name="biasMode">The desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Softmax(
            int outputs, 
            WeightsInitializationMode weightsMode = WeightsInitializationMode.GlorotUniform, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnSoftmaxLayer(input, outputs, weightsMode, biasMode);

        /// <summary>
        /// Creates a convolutional layer with the desired number of kernels
        /// </summary>
        /// <param name="info">The info on the convolution operation to perform</param>
        /// <param name="kernel">The volume information of the kernels used in the layer</param>
        /// <param name="kernels">The number of convolution kernels to apply to the input volume</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Convolutional(
            ConvolutionInfo info, (int X, int Y) kernel, int kernels, ActivationFunctionType activation,
            BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnConvolutionalLayer(input, info, kernel, kernels, activation, biasMode);

        /// <summary>
        /// Creates a pooling layer with a window of size 2 and a stride of 2
        /// </summary>
        /// <param name="info">The info on the pooling operation to perform</param>
        /// <param name="activation">The desired activation function to use in the network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Pooling(PoolingInfo info, ActivationFunctionType activation) => input => new CuDnnPoolingLayer(input, info, activation);

        /// <summary>
        /// Creates a new inception layer with the given features
        /// </summary>
        /// <param name="info">The info on the operations to execute inside the layer</param>
        /// <param name="biasMode">Indicates the desired initialization mode to use for the layer bias values</param>
        [PublicAPI]
        [Pure, NotNull]
        public static LayerFactory Inception(InceptionInfo info, BiasInitializationMode biasMode = BiasInitializationMode.Zero) 
            => input => new CuDnnInceptionLayer(input, info, biasMode);

        #region Feature helper

        /// <summary>
        /// A private class that is used to create a new standalone type that contains the actual test method (decoupling is needed to &lt;Module&gt; loading crashes)
        /// </summary>
        private static class CuDnnSupportHelper
        {
            /// <summary>
            /// Checks whether or not the Cuda features are currently supported
            /// </summary>
            public static bool IsGpuAccelerationSupported()
            {
                try
                {
                    // CUDA test
                    using (Alea.Gpu gpu = Alea.Gpu.Default)
                    {
                        if (gpu == null) return false;
                        if (!Alea.cuDNN.Dnn.IsAvailable) return false; // cuDNN
                        using (Alea.DeviceMemory<float> sample_gpu = gpu.AllocateDevice<float>(1024))
                        {
                            Alea.deviceptr<float> ptr = sample_gpu.Ptr;
                            void Kernel(int i) => ptr[i] = i;
                            Alea.Parallel.GpuExtension.For(gpu, 0, 1024, Kernel); // JIT test
                            float[] sample = Alea.Gpu.CopyToHost(sample_gpu);
                            return Enumerable.Range(0, 1024).Select<int, float>(i => i).ToArray().ContentEquals(sample);
                        }
                    }
                }
                catch
                {
                    // Missing .dll or other errors
                    return false;
                }
            }
        }

        #endregion
    }
}