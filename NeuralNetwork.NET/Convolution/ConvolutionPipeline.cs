using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

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
        public IReadOnlyList<ConvolutionsStackProcessor> Pipeline { get; }

        /// <summary>
        /// Initializes a new instance with the given pipeline
        /// </summary>
        /// <param name="pipeline">The convolution pipeline to execute</param>
        [PublicAPI]
        public ConvolutionPipeline([NotNull, ItemNotNull] IReadOnlyList<ConvolutionsStackProcessor> pipeline)
        {
            if (pipeline.Count == 0) throw new ArgumentOutOfRangeException("The pipeline must contain at least a function");
            Pipeline = pipeline;
        }

        /// <summary>
        /// Processes the input vector through the current pipeline
        /// </summary>
        /// <param name="input">The input to process</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public ConvolutionsStack Process([NotNull] double[,] input)
        {
            ConvolutionsStack result = ConvolutionsStack.From2DLayer(input);
            foreach (ConvolutionsStackProcessor p in Pipeline)
                result = p(result);
            return result;
        }

        /// <summary>
        /// Processes the input vector sthrough the current pipeline and returns a series of results
        /// </summary>
        /// <param name="inputs">The inputs to process</param>
        [PublicAPI]
        [Pure, NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public IReadOnlyList<ConvolutionsStack> Process([NotNull, ItemNotNull] IReadOnlyList<double[,]> inputs)
        {
            ConvolutionsStack[] results = new ConvolutionsStack[inputs.Count];
            bool result = ParallelCompatibilityWrapper.Instance.Invoke(0, inputs.Count, i => results[i] = Process(inputs[i]));
            if (!result) throw new Exception("Error executing the parallel loop");
            return results;
        }

        /// <summary>
        /// Flattens a vector of data volumes into a single 2D matrix
        /// </summary>
        /// <param name="data">The data to convert</param>
        [PublicAPI]
        [Pure]
        private static double[,] ConvertToMatrix([NotNull, ItemNotNull] params ConvolutionsStack[] data)
        {
            // Checks
            if (data.Length == 0) throw new ArgumentOutOfRangeException("The data array can't be empty");

            // Prepare the base network and the input data
            int
                depth = data[0].Depth, // Depth of each convolution volume
                ch = data[0].Height, // Height of each convolution layer
                cw = data[0].Width, // Width of each convolution layer
                lsize = ch * cw,
                volume = depth * lsize;

            // Additional checks
            if (data.Any(stack => stack.Depth != depth || stack.Height != ch || stack.Width != cw))
                throw new ArgumentException("The input data isn't coherent");

            // Setup the matrix with all the batched inputs
            double[,] x = new double[data.Length, volume];

            // Populate the matrix, iterate over all the volumes
            bool result = ParallelCompatibilityWrapper.Instance.Invoke(0, data.Length, i =>
            {
                unsafe
                {
                    // Fix the pointers
                    fixed (double* px = x)
                    {
                        ConvolutionsStack stack = data[i];
                        for (int j = 0; j < depth; j++) // Iterate over all the depth layer in each volume
                            for (int z = 0; z < ch; z++) // Height of each layer
                                for (int w = 0; w < cw; w++) // Width of each layer
                                    px[i * volume + j * lsize + z * ch + w] = stack[j, z, w];
                    }
                }
            });
            if (!result) throw new Exception("Error while running the parallel loop");
            return x;
        }
    }
}
