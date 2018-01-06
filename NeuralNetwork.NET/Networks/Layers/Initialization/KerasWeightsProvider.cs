using System;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Layers.Initialization
{
    /// <summary>
    /// A static class with some weights initialization methods ported over the Keras library
    /// </summary>
    internal static class KerasWeightsProvider
    {
        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the LeCun uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithLeCunUniform(in Tensor tensor, int fanIn)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            float scale = (float)Math.Sqrt(3f / fanIn);
            tensor.AsSpan().Fill(() => ThreadSafeRandom.NextUniform(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the Glorot &amp; Bengio normal distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        public static void FillWithGlorotNormal(in Tensor tensor, int fanIn, int fanOut)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanOut < 0) throw new ArgumentOutOfRangeException(nameof(fanOut), "The fan out must be a positive number");
            float scale = (float)Math.Sqrt(2f / (fanIn + fanOut));
            tensor.AsSpan().Fill(() => ThreadSafeRandom.NextGaussian(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the Glorot &amp; Bengio uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        public static void FillWithGlorotUniform(in Tensor tensor, int fanIn, int fanOut)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanOut < 0) throw new ArgumentOutOfRangeException(nameof(fanOut), "The fan out must be a positive number");
            float scale = (float)Math.Sqrt(6f / (fanIn + fanOut));
            tensor.AsSpan().Fill(() => ThreadSafeRandom.NextUniform(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the He et al. normal distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithHeEtAlNormal(in Tensor tensor, int fanIn)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            float scale = (float)Math.Sqrt(2f / fanIn);
            tensor.AsSpan().Fill(() => ThreadSafeRandom.NextGaussian(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the He et al. uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithHeEtAlUniform(in Tensor tensor, int fanIn)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            float scale = (float)Math.Sqrt(6f / fanIn);
            tensor.AsSpan().Fill(() => ThreadSafeRandom.NextUniform(scale));
        }
    }
}
