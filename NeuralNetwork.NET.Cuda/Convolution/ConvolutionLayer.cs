using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Cuda.Helpers;

namespace NeuralNetworkNET.Cuda.Convolution
{
    /// <summary>
    /// A class that represents a convolution pipeline with a series of sequential operations
    /// </summary>
    public sealed class ConvolutionPipeline2
    {
        /// <summary>
        /// Gets the pipeline in use in the current instance
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<ConvolutionOperation> Pipeline { get; }

        /// <summary>
        /// Initializes a new instance with the given pipeline
        /// </summary>
        /// <param name="pipeline">The convolution pipeline to execute</param>
        [PublicAPI]
        public ConvolutionPipeline2([NotNull, ItemNotNull] params ConvolutionOperation[] pipeline)
        {
            if (pipeline.Length == 0) throw new ArgumentOutOfRangeException("The pipeline must contain at least a function");
            Pipeline = pipeline;
        }

        /// <summary>
        /// Processes the input vector through the current pipeline
        /// </summary>
        /// <param name="input">The input to process</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[,] Process([NotNull] double[,] input)
        {
            int h = input.GetLength(0), w = input.GetLength(1);
            double[,] result = new double[h, w];
            Buffer.BlockCopy(input, 0, result, 0, sizeof(double) * w * h);
            int subdivision = 1;
            foreach (ConvolutionOperation operation in Pipeline)
            {
                switch (operation)
                {
                    case KernelConvolution k when k.OperationType == ConvolutionOperationType.Convolution3x3:
                      //  result = result.Convolute3x3(subdivision, (double[][,])k.Kernels);
                        subdivision *= k.Kernels.Count;
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Normalization:
                        result.Normalize(subdivision);
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Pool2x2:
                        result = result.Pool2x2(subdivision);
                        break;
                    case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.ReLU:
                        result.ReLU();
                        break;
                    default:
                        throw new ArgumentOutOfRangeException("Unsupported convolution operation");
                }
            }
            return result;
        }
    }

    public static class ConvolutionOperations
    {
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

    public class ConvolutionOperation
    {
        public ConvolutionOperationType OperationType { get; set; }

        public ConvolutionOperation(ConvolutionOperationType type) => OperationType = type;
    }

    public enum ConvolutionOperationType : byte
    {
        Convolution3x3,
        ReLU,
        Pool2x2,
        Normalization
    }

    class KernelConvolution : ConvolutionOperation
    {
        public IReadOnlyList<double[,]> Kernels { get; }

        public KernelConvolution(ConvolutionOperationType type, IReadOnlyList<double[,]> kernels) : base(type)
        {
            Kernels = kernels;
        }
    }
}
