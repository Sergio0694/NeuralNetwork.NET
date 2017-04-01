using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary.Convolution
{
    /// <summary>
    /// A class that represents a convolution pipeline with a series of sequential operations
    /// </summary>
    public class ConvolutionPipeline
    {
        /// <summary>
        /// Gets the pipeline in use in the current instance
        /// </summary>
        [NotNull]
        public Func<double[,], double[,]>[] Pipeline { get; }

        /// <summary>
        /// Initializes a new instance with the given pipeline
        /// </summary>
        /// <param name="pipeline">The convolution pipeline to execute</param>
        public ConvolutionPipeline([NotNull] Func<double[,], double[,]>[] pipeline)
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
        public double[,] Process([NotNull] double[,] input)
        {
            double[,] result = input;
            foreach (Func<double[,], double[,]> f in Pipeline)
                input = f(input);
            return result;
        }

        /// <summary>
        /// Processes the input vector sthrough the current pipeline and returns a series of results
        /// </summary>
        /// <param name="input">The inputs to process</param>
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public IList<double[,]> Process([NotNull] IList<double[,]> inputs)
        {
            double[][,] results = new double[inputs.Count][,];
            ParallelLoopResult result = Parallel.For(0, inputs.Count, i => results[i] = Process(inputs[i]));
            if (!result.IsCompleted) throw new Exception("Error executing the parallel loop");
            return results;
        }
    }
}
