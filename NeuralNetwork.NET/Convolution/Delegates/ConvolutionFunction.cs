using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution.Delegates
{
    /// <summary>
    /// A delegate that wraps a function to perform a kernel convolution on the input data
    /// </summary>
    /// <param name="data">The source data to process</param>
    /// <param name="kernel">The kernel to use</param>
    /// <returns>The processed data matrix. Its size depends on the function used and on the size of the kernel</returns>
    /// <remarks>This delegate is used to wrap the desired convolution function: it is possible to use one from the
    /// library or a custom one, that can work with a kernel of arbitrary size</remarks>
    [NotNull]
    public delegate double[,] ConvolutionFunction([NotNull] double[,] data, [NotNull] double[,] kernel);
}
