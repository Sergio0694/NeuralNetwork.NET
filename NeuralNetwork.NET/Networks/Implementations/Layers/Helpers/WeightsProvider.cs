using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
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
        public static float[,] NewFullyConnectedWeights(int inputs, int outputs, WeightsInitializationMode mode)
        {
            if (inputs <= 0 || outputs <= 0) throw new ArgumentOutOfRangeException("The inputs and outputs must be positive numbers");
            switch (mode)
            {
                case WeightsInitializationMode.LeCunUniform: return ThreadSafeRandom.NextLeCunUniformMatrix(inputs, outputs);
                case WeightsInitializationMode.GlorotNormal: return ThreadSafeRandom.NextGlorotNormalMatrix(inputs, outputs);
                case WeightsInitializationMode.GlorotUniform: return ThreadSafeRandom.NextGlorotUniformMatrix(inputs, outputs);
                case WeightsInitializationMode.HeEtAlNormal: return ThreadSafeRandom.NextHeEtAlNormalMatrix(inputs, outputs);
                case WeightsInitializationMode.HeEtAlUniform: return ThreadSafeRandom.NextHeEtAlUniformMatrix(inputs, outputs);
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported weights initialization mode");
            }
        }

        /// <summary>
        /// Creates a weight matrix for a convolutional layer
        /// </summary>
        /// <param name="inputDepth">The depth of the input volume</param>
        /// <param name="kernelsHeight">The height of each kernel</param>
        /// <param name="kernelsWidth">The width of each kernel</param>
        /// <param name="kernels">The number of kernels in the layer</param>
        [Pure, NotNull]
        public static float[,] NewConvolutionalKernels(int inputDepth, int kernelsHeight, int kernelsWidth, int kernels)
        {
            if (kernels <= 0) throw new ArgumentOutOfRangeException(nameof(kernels), "The number of kernels must be positive");
            float scale = (float)Math.Sqrt(6f / (inputDepth * kernelsHeight * kernelsWidth));
            return ThreadSafeRandom.NextUniformMatrix(kernels, kernelsHeight * kernelsWidth * inputDepth, scale);
        }

        /// <summary>
        /// Creates a vector of biases for a network layer
        /// </summary>
        /// <param name="length">The length of the vector</param>
        /// <param name="mode">The initialization mode for the bias values</param>
        [Pure, NotNull]
        public static float[] NewBiases(int length, BiasInitializationMode mode)
        {
            if (length <= 0) throw new ArgumentException(nameof(length), "The biases vector must have a positive number of items");
            switch (mode)
            {
                case BiasInitializationMode.Zero: return new float[length];
                case BiasInitializationMode.Gaussian: return ThreadSafeRandom.NextGaussianVector(length);
                default: throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported biases initialization mode");
            }
        }
    }
}
