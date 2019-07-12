using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Core.APIs.Enums;
using NeuralNetworkDotNet.Core.APIs.Models;
using NeuralNetworkDotNet.Core.Helpers;

namespace NeuralNetworkDotNet.Core.Network.Initialization
{
    /// <summary>
    /// A simple class that provides some helper methods to quickly initialize the weights of a neural network layer
    /// </summary>
    internal static class WeightsProvider
    {
        /// <summary>
        /// Creates a weights vector for a fully connected layer
        /// </summary>
        /// <param name="inputs">The layer inputs</param>
        /// <param name="outputs">The output neurons</param>
        /// <param name="mode">The initialization mode for the weights</param>
        [Pure, NotNull]
        public static Tensor NewFullyConnectedWeights(int inputs, int outputs, WeightsInitializationMode mode)
        {
            Guard.IsTrue(inputs >= 0, nameof(inputs), "The inputs must be a positive number");
            Guard.IsTrue(outputs >= 0, nameof(outputs), "The outputs must be a positive number");

            var weights = Tensor.New(inputs, outputs);

            switch (mode)
            {
                case WeightsInitializationMode.LeCunUniform: KerasWeightsProvider.FillWithLeCunUniform(weights, weights.N); break;
                case WeightsInitializationMode.GlorotNormal: KerasWeightsProvider.FillWithGlorotNormal(weights, weights.N, weights.W); break;
                case WeightsInitializationMode.GlorotUniform: KerasWeightsProvider.FillWithGlorotUniform(weights, weights.N, weights.W); break;
                case WeightsInitializationMode.HeEtAlNormal: KerasWeightsProvider.FillWithHeEtAlNormal(weights, weights.N); break;
                case WeightsInitializationMode.HeEtAlUniform: KerasWeightsProvider.FillWithHeEtAlUniform(weights, weights.N); break;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported weights initialization mode");
            }

            return weights;
        }

        /// <summary>
        /// Creates a weights vector for a convolutional layer
        /// </summary>
        /// <param name="channels">The number of input channels</param>
        /// <param name="kernelsHeight">The height of each kernel</param>
        /// <param name="kernelsWidth">The width of each kernel</param>
        /// <param name="kernels">The number of kernels in the layer</param>
        [Pure, NotNull]
        public static Tensor NewConvolutionalKernels(int channels, int kernelsHeight, int kernelsWidth, int kernels)
        {
            Guard.IsTrue(channels >= 0, nameof(channels), "The input channels must be a positive number");
            Guard.IsTrue(kernelsHeight >= 0, nameof(kernelsHeight), "The height of the kernels must be a positive number");
            Guard.IsTrue(kernelsWidth >= 0, nameof(kernelsWidth), "The width of the kernels must be a positive number");
            Guard.IsTrue(kernels >= 0, nameof(kernels), "The number of kernels must be a positive number");

            var weights = Tensor.New(kernels, channels, kernelsHeight, kernelsWidth);
            KerasWeightsProvider.FillWithHeEtAlUniform(weights, weights.CHW);

            return weights;
        }

        /// <summary>
        /// Creates a vector of biases for a network layer
        /// </summary>
        /// <param name="length">The length of the vector</param>
        /// <param name="mode">The initialization mode for the bias values</param>
        [Pure, NotNull]
        public static Tensor NewBiases(int length, BiasInitializationMode mode)
        {
            Guard.IsFalse(length <= 0, nameof(length), "The biases vector must have a positive number of items");

            var biases = Tensor.New(1, length);

            switch (mode)
            {
                case BiasInitializationMode.Zero: biases.Span.Clear(); break;
                case BiasInitializationMode.Gaussian: biases.Span.Fill(() => ConcurrentRandom.Instance.NextGaussian()); return biases;
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported biases initialization mode");
            }

            return biases;
        }

        /// <summary>
        /// Creates a new weights vector for a batch normalization layer
        /// </summary>
        /// <param name="c">The C dimension of the input tensor</param>
        /// <param name="hw">The HW dimension of the input tensor</param>
        /// <param name="mode">The desired normalization mode to use</param>
        [Pure, NotNull]
        public static Tensor NewGammaParameters(int c, int hw, NormalizationMode mode)
        {
            int l;
            if (mode == NormalizationMode.Spatial) l = c;
            else if (mode == NormalizationMode.PerActivation) l = c * hw;
            else throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");

            var weights = Tensor.New(1, l);

            ref var rw = ref weights.Span.GetPinnableReference();
            for (var i = 0; i < l; i++)
                Unsafe.Add(ref rw, i) = 1;

            return weights;
        }

        /// <summary>
        /// Creates a new beta weights vector for a batch normalization layer
        /// </summary>
        /// <param name="c">The C dimension of the input tensor</param>
        /// <param name="hw">The HW dimension of the input tensor</param>
        /// <param name="mode">The normalization mode to use</param>
        [Pure, NotNull]
        public static Tensor NewBetaParameters(int c, int hw, NormalizationMode mode)
        {
            switch (mode)
            {
                case NormalizationMode.Spatial: return NewBiases(c, BiasInitializationMode.Zero);
                case NormalizationMode.PerActivation: return NewBiases(c * hw, BiasInitializationMode.Zero);
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Invalid normalization mode");
            }
        }
    }
}
