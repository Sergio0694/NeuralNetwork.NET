using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution.Operations
{
    /// <summary>
    /// A class that represents a kernel convolution operation
    /// </summary>
    public sealed class KernelConvolution : ConvolutionOperation
    {
        /// <summary>
        /// Gets the kernels chosen for the current instance
        /// </summary>
        [NotNull]
        public double[][,] Kernels { get; }

        /// <summary>
        /// Creates a new instance with the given kernels and type
        /// </summary>
        /// <param name="type">The type of operation</param>
        /// <param name="kernels">The kernels to use</param>
        internal KernelConvolution(ConvolutionOperationType type, [NotNull] double[][,] kernels) : base(type)
        {
            if (type != ConvolutionOperationType.Convolution3x3) throw new ArgumentOutOfRangeException("Invalid operation type");
            Kernels = kernels;
        }
    }
}
