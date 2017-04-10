using JetBrains.Annotations;

namespace NeuralNetworkNET.Convolution.Delegates
{
    /// <summary>
    /// A delegate that processes a volume of data (a stack of rectangular matrices) and returns a new data volume
    /// </summary>
    /// <param name="stack">The data volume (width*height*depth) to process, layer by layer</param>
    [NotNull]
    public delegate ConvolutionsStack ConvolutionsStackProcessor([NotNull] ConvolutionsStack stack);
}
