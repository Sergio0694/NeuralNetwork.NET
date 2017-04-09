using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetwork.NET.Convolution
{
    /// <summary>
    /// A delegate that processes a volume of data (a stack of rectangular matrices) and returns a new data volume
    /// </summary>
    /// <param name="volume">The data volume (width*height*depth) to process, layer by layer</param>
    public delegate double[][,] VolumicProcessor([NotNull] double[][,] volume);

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
        public double[][,] Process([NotNull] double[,] input)
        {
            double[][,] result = { input };
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
        public IReadOnlyList<double[][,]> Process([NotNull] IReadOnlyList<double[,]> inputs)
        {
            double[][][,] results = new double[inputs.Count][][,];
            bool result = ParallelCompatibilityWrapper.Instance.Invoke(0, inputs.Count, i => results[i] = Process(inputs[i]));
            if (!result) throw new Exception("Error executing the parallel loop");
            return results;
        }
    }
}
