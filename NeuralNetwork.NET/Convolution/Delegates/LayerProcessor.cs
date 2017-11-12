using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution.Delegates
{
    /// <summary>
    /// A delegate that wraps a function that processes a single data layer (with normalization or another operation)
    /// </summary>
    /// <param name="data">The source data</param>
    /// <remarks>The resulting data matrix should have the same size of the original</remarks>
    [NotNull]
    internal delegate float[,] LayerProcessor([NotNull] float[,] data);
}
