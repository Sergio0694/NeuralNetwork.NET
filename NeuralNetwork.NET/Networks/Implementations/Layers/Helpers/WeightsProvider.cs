using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Helpers
{
    /// <summary>
    /// A simple class that provides some helper methods to quickly initialize the weights of a neural network layer
    /// </summary>
    internal static class WeightsProvider
    {
        /// <summary>
        /// Creates a weight matrix for a fully connected layer
        /// </summary>
        /// <param name="inputs">The input neurons</param>
        /// <param name="outputs">The output neurons</param>
        /// <param name="mode">The initialization mode for the weights</param>
        [Pure, NotNull]
        public static unsafe float[] NewFullyConnectedWeights(int inputs, int outputs, WeightsInitializationMode mode)
        {
            if (inputs <= 0 || outputs <= 0) throw new ArgumentOutOfRangeException("The inputs and outputs must be positive numbers");
            float[] weights = new float[inputs * outputs];
            fixed (float* pw = weights)
            {
                Tensor.Reshape(pw, inputs, outputs, out Tensor wTensor);
                switch (mode)
                {
                    case WeightsInitializationMode.LeCunUniform:
                        KerasWeightsProvider.FillWithLeCunUniform(wTensor, inputs);
                        break;
                    case WeightsInitializationMode.GlorotNormal:
                        KerasWeightsProvider.FillWithGlorotNormal(wTensor, inputs, outputs);
                        break;
                    case WeightsInitializationMode.GlorotUniform:
                        KerasWeightsProvider.FillWithGlorotUniform(wTensor, inputs, outputs);
                        break;
                    case WeightsInitializationMode.HeEtAlNormal:
                        KerasWeightsProvider.FillWithHeEtAlNormal(wTensor, inputs);
                        break;
                    case WeightsInitializationMode.HeEtAlUniform:
                        KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, inputs);
                        break;
                    default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported weights initialization mode");
                }
            }
            return weights;
        }

        /// <summary>
        /// Creates a weight matrix for a convolutional layer
        /// </summary>
        /// <param name="inputDepth">The depth of the input volume</param>
        /// <param name="kernelsHeight">The height of each kernel</param>
        /// <param name="kernelsWidth">The width of each kernel</param>
        /// <param name="kernels">The number of kernels in the layer</param>
        [Pure, NotNull]
        public static unsafe float[] NewConvolutionalKernels(int inputDepth, int kernelsHeight, int kernelsWidth, int kernels)
        {
            if (kernels <= 0) throw new ArgumentOutOfRangeException(nameof(kernels), "The number of kernels must be positive");
            float[] weights = new float[kernels * kernelsHeight * kernelsWidth * inputDepth];
            fixed (float* pw = weights)
            {
                Tensor.Reshape(pw, 1, weights.Length, out Tensor wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, inputDepth * kernelsHeight * kernelsWidth);
            }
            return weights;
        }

        /// <summary>
        /// Creates a vector of biases for a network layer
        /// </summary>
        /// <param name="length">The length of the vector</param>
        /// <param name="mode">The initialization mode for the bias values</param>
        [Pure, NotNull]
        public static unsafe float[] NewBiases(int length, BiasInitializationMode mode)
        {
            if (length <= 0) throw new ArgumentException(nameof(length), "The biases vector must have a positive number of items");
            float[] biases = new float[length];
            switch (mode)
            {
                case BiasInitializationMode.Zero: return biases;
                case BiasInitializationMode.Gaussian:
                    fixed (float* pb = biases)
                    {
                        Tensor.Reshape(pb, 1, length, out Tensor bTensor);
                        bTensor.Fill(() => ThreadSafeRandom.NextGaussian());
                        return biases;
                    }
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported biases initialization mode");
            }
        }
    }
}
