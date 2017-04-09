using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution
{
    /// <summary>
    /// A delegate that processes a volume of data (a stack of rectangular matrices) and returns a new data volume
    /// </summary>
    /// <param name="volume">The data volume (width*height*depth) to process, layer by layer</param>
    public delegate double[,,] VolumicProcessor([NotNull] double[,,] volume);

    /// <summary>
    /// A class that represents a convolution pipeline with a series of sequential operations
    /// </summary>
    public sealed class ConvolutionPipeline
    {
        /// <summary>
        /// Gets the pipeline in use in the current instance
        /// </summary>
        [NotNull]
        public VolumicProcessor[] Pipeline { get; }

        /// <summary>
        /// Initializes a new instance with the given pipeline
        /// </summary>
        /// <param name="pipeline">The convolution pipeline to execute</param>
        [PublicAPI]
        public ConvolutionPipeline([NotNull] VolumicProcessor[] pipeline)
        {
            if (pipeline.Length == 0) throw new ArgumentOutOfRangeException("The pipeline must contain at least a function");
            Pipeline = pipeline;
        }

        /// <summary>
        /// Processes the input vector through the current pipeline
        /// </summary>
        /// <param name="input">The input to process</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[,,] Process([NotNull] double[,] input)
        {
            double[,,] result = new double[1, input.GetLength(0), input.GetLength(1)];
            foreach (VolumicProcessor p in Pipeline)
                result = p(result);
            return result;
        }

        /// <summary>
        /// Processes the input vector sthrough the current pipeline and returns a series of results
        /// </summary>
        /// <param name="inputs">The inputs to process</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public IReadOnlyList<double[,,]> Process([NotNull] IReadOnlyList<double[,]> inputs)
        {
            double[][,,] results = new double[inputs.Count][,,];
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
        [NotNull]
        private static double[,] ConvertToMatrix([NotNull] params double[][,,] data)
        {
            // Checks
            if (data.Length == 0) throw new ArgumentOutOfRangeException("The data array can't be empty");

            // Prepare the base network and the input data
            int
                depth = data[0].GetLength(0), // Depth of each convolution volume
                ch = data[0].GetLength(1), // Height of each convolution layer
                cw = data[0].GetLength(2), // Width of each convolution layer
                lsize = ch * cw,
                volume = depth * lsize;
            double[,] x = new double[data.Length, volume]; // Matrix with all the batched inputs

            // Populate the matrix, iterate over all the volumes
            bool result = ParallelCompatibilityWrapper.Instance.Invoke(0, data.Length, i =>
            {
                unsafe
                {
                    // Fix the pointers
                    fixed (double* px = x, pdi = data[i])
                    {
                        for (int j = 0; j < depth; j++) // Iterate over all the depth layer in each volume
                            for (int z = 0; z < ch; z++) // Height of each layer
                                for (int w = 0; w < cw; w++) // Width of each layer
                                    px[i * volume + j * lsize + z * ch + w] = pdi[j * lsize + z * cw + w];
                    }
                }
            });
            if (!result) throw new Exception("Error while running the parallel loop");
            return x;
        }
    }
}
