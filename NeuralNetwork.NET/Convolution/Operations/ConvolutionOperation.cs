using System;
using System.Linq;

namespace NeuralNetworkNET.Convolution.Operations
{
    public class ConvolutionOperation
    {
        public ConvolutionOperationType OperationType { get; set; }

        protected internal ConvolutionOperation(ConvolutionOperationType type) => OperationType = type;

        public static ConvolutionOperation ReLU { get; } = new ConvolutionOperation(ConvolutionOperationType.ReLU);

        public static ConvolutionOperation Pool2x2 { get; } = new ConvolutionOperation(ConvolutionOperationType.Pool2x2);

        public static ConvolutionOperation Normalization { get; } = new ConvolutionOperation(ConvolutionOperationType.Normalization);

        public static ConvolutionOperation Convolution3x3(params double[][,] kernels)
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
