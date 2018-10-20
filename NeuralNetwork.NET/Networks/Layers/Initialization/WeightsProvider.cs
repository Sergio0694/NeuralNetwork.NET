using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Layers.Initialization
{
    /// <summary>
    /// A simple class that provides some helper methods to quickly initialize the weights of a neural network layer
    /// </summary>
    internal static class WeightsProvider
    {
        /// <summary>
        /// Creates a weights vector for a fully connected layer
        /// </summary>
        /// <param name="input">The layer inputs</param>
        /// <param name="outputs">The output neurons</param>
        /// <param name="mode">The initialization mode for the weights</param>
        [Pure, NotNull]
        public static unsafe float[] NewFullyConnectedWeights(in TensorInfo input, int outputs, WeightsInitializationMode mode)
        {
            float[] weights = new float[input.Size * outputs];
            fixed (float* pw = weights)
            {
                Tensor.Reshape(pw, input.Size, outputs, out Tensor wTensor);
                switch (mode)
                {
                    case WeightsInitializationMode.LeCunUniform:
                        KerasWeightsProvider.FillWithLeCunUniform(wTensor, input.Size);
                        break;
                    case WeightsInitializationMode.GlorotNormal:
                        KerasWeightsProvider.FillWithGlorotNormal(wTensor, input.Size, outputs);
                        break;
                    case WeightsInitializationMode.GlorotUniform:
                        KerasWeightsProvider.FillWithGlorotUniform(wTensor, input.Size, outputs);
                        break;
                    case WeightsInitializationMode.HeEtAlNormal:
                        KerasWeightsProvider.FillWithHeEtAlNormal(wTensor, input.Size);
                        break;
                    case WeightsInitializationMode.HeEtAlUniform:
                        KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Size);
                        break;
                    default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported weights initialization mode");
                }
            }
            return weights;
        }

        /// <summary>
        /// Creates a weights vector for a convolutional layer
        /// </summary>
        /// <param name="input">The layer inputs</param>
        /// <param name="kernelsHeight">The height of each kernel</param>
        /// <param name="kernelsWidth">The width of each kernel</param>
        /// <param name="kernels">The number of kernels in the layer</param>
        [Pure, NotNull]
        public static unsafe float[] NewConvolutionalKernels(in TensorInfo input, int kernelsHeight, int kernelsWidth, int kernels)
        {
            if (kernels <= 0) throw new ArgumentOutOfRangeException(nameof(kernels), "The number of kernels must be positive");
            float[] weights = new float[kernels * kernelsHeight * kernelsWidth * input.Channels];
            fixed (float* pw = weights)
            {
                Tensor.Reshape(pw, 1, weights.Length, out Tensor wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Channels * kernelsHeight * kernelsWidth);
            }
            return weights;
        }

        /// <summary>
        /// Creates a new mixed weights vector for an inception layer
        /// </summary>
        /// <param name="input">The layer inputs</param>
        /// <param name="info">The info on the target inception layer</param>
        [Pure, NotNull]
        public static unsafe float[] NewInceptionWeights(in TensorInfo input, in InceptionInfo info)
        {
            // Setup
            int
                _1x1Length = input.Channels * info.Primary1x1ConvolutionKernels,
                _3x3Reduce1x1Length = input.Channels * info.Primary3x3Reduce1x1ConvolutionKernels,
                _3x3Length = 3 * 3 * info.Primary3x3Reduce1x1ConvolutionKernels * info.Secondary3x3ConvolutionKernels,
                _5x5Reduce1x1Length = input.Channels * info.Primary5x5Reduce1x1ConvolutionKernels,
                _5x5Length = 5 * 5 * info.Primary5x5Reduce1x1ConvolutionKernels * info.Secondary5x5ConvolutionKernels,
                secondary1x1Length = input.Channels * info.Secondary1x1AfterPoolingConvolutionKernels;
            float[] weights = new float[_1x1Length + _3x3Reduce1x1Length + _3x3Length + _5x5Reduce1x1Length + _5x5Length + secondary1x1Length];
            fixed (float* pw = weights)
            {
                // 1x1
                Tensor.Reshape(pw, 1, _1x1Length, out Tensor wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Channels);

                // 3x3 reduce 1x1
                Tensor.Reshape(pw + _1x1Length, 1, _3x3Reduce1x1Length, out wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Channels);

                // 3x3
                Tensor.Reshape(pw + _1x1Length + _3x3Reduce1x1Length, 1, _3x3Length, out wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, 3 * 3 * info.Primary3x3Reduce1x1ConvolutionKernels);

                // 5x5 reduce 1x1
                Tensor.Reshape(pw + _1x1Length + _3x3Reduce1x1Length + _3x3Length, 1, _5x5Reduce1x1Length, out wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Channels);

                // 5x5
                Tensor.Reshape(pw + _1x1Length + _3x3Reduce1x1Length + _3x3Length + _5x5Reduce1x1Length, 1, _5x5Length, out wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, 5 * 5 * info.Primary5x5Reduce1x1ConvolutionKernels);

                // Pool 1x1
                Tensor.Reshape(pw + _1x1Length + _3x3Reduce1x1Length + _3x3Length + _5x5Reduce1x1Length + _5x5Length, 1, secondary1x1Length, out wTensor);
                KerasWeightsProvider.FillWithHeEtAlUniform(wTensor, input.Channels);
            }
            return weights;
        }

        /// <summary>
        /// Creates a vector of biases for a network layer
        /// </summary>
        /// <param name="length">The length of the vector</param>
        /// <param name="mode">The initialization mode for the bias values</param>
        [Pure, NotNull]
        public static float[] NewBiases(int length, BiasInitializationMode mode)
        {
            if (length <= 0) throw new ArgumentException("The biases vector must have a positive number of items", nameof(length));
            float[] biases = new float[length];
            switch (mode)
            {
                case BiasInitializationMode.Zero: return biases;
                case BiasInitializationMode.Gaussian:
                    biases.AsSpan().Fill(() => ThreadSafeRandom.NextGaussian());
                    return biases;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported biases initialization mode");
            }
        }

        /// <summary>
        /// Creates a new weights vector for a batch normalization layer
        /// </summary>
        /// <param name="shape">The layer inputs and ouputs</param>
        /// <param name="mode">The normalization mode to use</param>
        [Pure, NotNull]
        public static unsafe float[] NewGammaParameters(in TensorInfo shape, NormalizationMode mode)
        {
            int l;
            if (mode == NormalizationMode.Spatial) l = shape.Channels;
            else if (mode == NormalizationMode.PerActivation) l = shape.Size;
            else throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            float[] weights = new float[l];
            fixed (float* pw = weights)
                for (int i = 0; i < l; i++)
                    pw[i] = 1;
            return weights;
        }

        /// <summary>
        /// Creates a new beta weights vector for a batch normalization layer
        /// </summary>
        /// <param name="shape">The layer inputs and ouputs</param>
        /// <param name="mode">The normalization mode to use</param>
        [Pure, NotNull]
        public static float[] NewBetaParameters(in TensorInfo shape, NormalizationMode mode)
        {
            switch (mode)
            {
                case NormalizationMode.Spatial: return NewBiases(shape.Channels, BiasInitializationMode.Zero);
                case NormalizationMode.PerActivation: return NewBiases(shape.Size, BiasInitializationMode.Zero);
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }
    }
}
