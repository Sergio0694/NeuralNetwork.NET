using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Operations;

namespace NeuralNetworkNET.Convolution.Delegates
{
    /// <summary>
    /// A delegate that performs executes a convolution pipeline and processes the given input
    /// </summary>
    /// <param name="pipeline">The list of operations to perform</param>
    /// <param name="input">The input to process</param>
    [NotNull]
    internal delegate float[,] ConvolutionPipelineProcessor([NotNull] IReadOnlyList<ConvolutionOperation> pipeline, [NotNull] float[,] input);
}
