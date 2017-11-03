using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Operations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Convolution
{
    /// <summary>
    /// A class that executes a convolution pipeline using GPU parallelization
    /// </summary>
    public static class ConvolutionGpuPipeline
    {
        /// <summary>
        /// Processes the input maatrix through the current pipeline
        /// </summary>
        /// <param name="pipeline">The list of operations to perform</param>
        /// <param name="input">The input to process</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Process([NotNull] IReadOnlyList<ConvolutionOperation> pipeline, [NotNull] double[,] input)
        {
            int h = input.GetLength(0), w = input.GetLength(1);
            double[,] copy = new double[h, w];
            Buffer.BlockCopy(input, 0, copy, 0, sizeof(double) * w * h);
            double[][,] result = { copy };
            int subdivision = 1;
            foreach (ConvolutionOperation operation in pipeline)
            {
                switch (operation)
                {
                    case KernelConvolution k when k.OperationType == ConvolutionOperationType.Convolution3x3:
                        List<double[,]> convolutions = new List<double[,]>();
                        for (int i = 0; i < result.Length; i++)
                        {
                            double[][,] partial = result[i].Convolute3x3(subdivision, k.Kernels);
                            convolutions.AddRange(partial);
                        }
                        result = convolutions.ToArray();
                        subdivision *= k.Kernels.Length;
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Normalization:
                        foreach (double[,] m in result) m.Normalize(subdivision);
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Pool2x2:
                        for (int i = 0; i < result.Length; i++) result[i] = result[i].Pool2x2(subdivision);
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.ReLU:
                        foreach (double[,] m in result) m.ReLU();
                        break;
                    default:
                        throw new ArgumentOutOfRangeException("Unsupported convolution operation");
                }
            }
            return result.MergeRows();
        }
    }
}
