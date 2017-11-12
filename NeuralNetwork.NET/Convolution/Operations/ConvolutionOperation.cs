using System;
using System.Linq;

namespace NeuralNetworkNET.Convolution.Operations
{
    /// <summary>
    /// A class that represents a convolution operation to perform in a pipeline
    /// </summary>
    public class ConvolutionOperation
    {
        /// <summary>
        /// Gets the operation type for the current instance
        /// </summary>
        public ConvolutionOperationType OperationType { get; }

        /// <summary>
        /// Creates a new instance that wraps the given operation
        /// </summary>
        /// <param name="type">The type of operation to perform</param>
        protected internal ConvolutionOperation(ConvolutionOperationType type) => OperationType = type;

        /// <summary>
        /// Gets the ReLU operation instance
        /// </summary>
        public static ConvolutionOperation ReLU { get; } = new ConvolutionOperation(ConvolutionOperationType.ReLU);

        /// <summary>
        /// Gets the 2x2 pooling operation instance
        /// </summary>
        public static ConvolutionOperation Pool2x2 { get; } = new ConvolutionOperation(ConvolutionOperationType.Pool2x2);

        /// <summary>
        /// Gets the normalization operation instance
        /// </summary>
        public static ConvolutionOperation Normalization { get; } = new ConvolutionOperation(ConvolutionOperationType.Normalization);

        /// <summary>
        /// Creates a new instance that represents a 3x3 convolution with the given kernels
        /// </summary>
        /// <param name="kernels">The kernels to use to perform the convolution</param>
        public static ConvolutionOperation Convolution3x3(params float[][,] kernels)
        {
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels list can't be empty");
            int h = kernels[0].GetLength(0), w = kernels[0].GetLength(1);
            if (h != 3 || w != 3) throw new ArgumentException(nameof(kernels), "The size of the kernels must be 3x3");
            if (kernels.Skip(1).Any(k => k.GetLength(0) != h || k.GetLength(1) != w))
                throw new ArgumentException(nameof(kernels), "The size of all the kernels must be the same");
            return new KernelConvolution(ConvolutionOperationType.Convolution3x3, kernels);
        }
    }
}
