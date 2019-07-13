using System;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.Network.Initialization
{
    /// <summary>
    /// A <see langword="class"/> with some weights initialization methods ported over the Keras library
    /// </summary>
    internal static class KerasWeightsProvider
    {
        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the LeCun uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithLeCunUniform([NotNull] Tensor tensor, int fanIn)
        {
            Guard.IsFalse(fanIn < 0, nameof(fanIn), "The fan in must be a positive number");

            var scale = (float)Math.Sqrt(3f / fanIn);
            tensor.Span.Fill(() => ConcurrentRandom.Instance.NextUniform(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the Glorot &amp; Bengio normal distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        public static void FillWithGlorotNormal([NotNull] Tensor tensor, int fanIn, int fanOut)
        {
            Guard.IsFalse(fanIn < 0, nameof(fanIn), "The fan in must be a positive number");
            Guard.IsFalse(fanOut < 0, nameof(fanOut), "The fan out must be a positive number");

            var scale = (float)Math.Sqrt(2f / (fanIn + fanOut));
            tensor.Span.Fill(() => ConcurrentRandom.Instance.NextGaussian(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the Glorot &amp; Bengio uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        public static void FillWithGlorotUniform([NotNull] Tensor tensor, int fanIn, int fanOut)
        {
            Guard.IsFalse(fanIn < 0, nameof(fanIn), "The fan in must be a positive number");
            Guard.IsFalse(fanOut < 0, nameof(fanOut), "The fan out must be a positive number");

            var scale = (float)Math.Sqrt(6f / (fanIn + fanOut));
            tensor.Span.Fill(() => ConcurrentRandom.Instance.NextUniform(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the He et al. normal distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithHeEtAlNormal([NotNull] Tensor tensor, int fanIn)
        {
            Guard.IsFalse(fanIn < 0, nameof(fanIn), "The fan in must be a positive number");

            var scale = (float)Math.Sqrt(2f / fanIn);
            tensor.Span.Fill(() => ConcurrentRandom.Instance.NextGaussian(scale));
        }

        /// <summary>
        /// Fills the target <see cref="Tensor"/> with values from the He et al. uniform distribution
        /// </summary>
        /// <param name="tensor">The target <see cref="Tensor"/> to fill</param>
        /// <param name="fanIn">The input neurons</param>
        public static void FillWithHeEtAlUniform([NotNull] Tensor tensor, int fanIn)
        {
            Guard.IsFalse(fanIn < 0, nameof(fanIn), "The fan in must be a positive number");

            var scale = (float)Math.Sqrt(6f / fanIn);
            tensor.Span.Fill(() => ConcurrentRandom.Instance.NextUniform(scale));
        }
    }
}
