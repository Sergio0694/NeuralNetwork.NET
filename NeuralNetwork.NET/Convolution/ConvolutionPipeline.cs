using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Delegates;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Convolution.Operations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Convolution
{
    /// <summary>
    /// A class that represents a convolution pipeline with a series of sequential operations
    /// </summary>
    public sealed class ConvolutionPipeline
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
        public ConvolutionPipeline([NotNull, ItemNotNull] params ConvolutionOperation[] pipeline)
        {
            if (pipeline.Length == 0) throw new ArgumentOutOfRangeException("The pipeline must contain at least a function");
            Pipeline = pipeline;
        }

        /// <summary>
        /// Gets or sets an injected function that executes the pipeline processing
        /// </summary>
        [CanBeNull]
        internal static ConvolutionPipelineProcessor ProcessOverride { get; set; }

        /// <summary>
        /// Processes the input dataset through the current pipeline
        /// </summary>
        /// <param name="input">The dataset to process</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public float[,] Process([NotNull] float[,] input)
        {
            // Execute the override, if present
            if (ProcessOverride != null) return ProcessOverride(Pipeline, input);

            // Local function that processes a 2D layer
            ConvolutionsStack ProcessLayer(float[,] layer)
            {
                ConvolutionsStack processed = ConvolutionsStack.From2DLayer(layer);
                foreach (ConvolutionOperation operation in Pipeline)
                {
                    switch (operation)
                    {
                        case KernelConvolution k when k.OperationType == ConvolutionOperationType.Convolution3x3:
                            processed = processed.Expand(ConvolutionExtensions.Convolute3x3, k.Kernels);
                            break;
                        case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Normalization:
                            processed = processed.Process(ConvolutionExtensions.Normalize);
                            break;
                        case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.Pool2x2:
                            processed = processed.Process(ConvolutionExtensions.Pool2x2);
                            break;
                        case ConvolutionOperation op when op.OperationType == ConvolutionOperationType.ReLU:
                            processed = processed.Process(ConvolutionExtensions.ReLU);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException("Unsupported convolution operation");
                    }
                }
                return processed;
            }

            // Process the whole dataset in parallel
            IReadOnlyList<float[,]> volume = input.Extract3DVolume();
            ConvolutionsStack[] results = new ConvolutionsStack[volume.Count];
            bool result = Parallel.For(0, volume.Count, i => results[i] = ProcessLayer(volume[i])).IsCompleted;
            if (!result) throw new Exception("Error executing the parallel loop");
            return ConvolutionsStack.ConvertToMatrix(results);
        }
    }
}
