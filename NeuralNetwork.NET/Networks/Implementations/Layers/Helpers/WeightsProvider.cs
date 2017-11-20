using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;

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
        [Pure, NotNull]
        public static float[,] FullyConnectedWeights(int inputs, int outputs)
        {
            if (inputs <= 0 || outputs <= 0) throw new ArgumentOutOfRangeException("The inputs and outputs must be positive numbers");
            Random random = new Random();
            return random.NextXavierMatrix(inputs, outputs);
        }

        /// <summary>
        /// Creates a weight matrix for a convolutional layer
        /// </summary>
        /// <param name="axis">The 2D axis of each kernel</param>
        /// <param name="depth">The depth of each kernel volume</param>
        /// <param name="kernels">The number of kernels in the layer</param>
        [Pure, NotNull]
        public static float[,] ConvolutionalKernels(int axis, int depth, int kernels)
        {
            if (kernels <= 0) throw new ArgumentOutOfRangeException(nameof(kernels), "The number of kernels must be positive");
            Random random = new Random();
            int weights = axis * axis * depth;
            return random.NextUniformMatrix(kernels, weights, weights);
        }

        /// <summary>
        /// Creates a vector of biases for a network layer
        /// </summary>
        /// <param name="length">The length of the vector</param>
        [Pure, NotNull]
        public static float[] Biases(int length)
        {
            if (length <= 0) throw new ArgumentException(nameof(length), "The biases vector must have a positive number of items");
            Random random = new Random();
            return random.NextGaussianVector(length);
        }
    }
}
