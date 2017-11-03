using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution.Operations
{
    public sealed class KernelConvolution : ConvolutionOperation
    {
        [NotNull]
        public double[][,] Kernels { get; }

        internal KernelConvolution(ConvolutionOperationType type, [NotNull] double[][,] kernels) : base(type)
        {
            Kernels = kernels;
        }
    }
}
