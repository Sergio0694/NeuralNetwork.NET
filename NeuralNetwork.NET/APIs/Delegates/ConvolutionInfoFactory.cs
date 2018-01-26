using NeuralNetworkNET.APIs.Structs;

namespace NeuralNetworkNET.APIs.Delegates
{
    /// <summary>
    /// A <see langword="delegate"/> that calculates a <see cref="ConvolutionInfo"/> value from the input <see cref="TensorInfo"/> instances
    /// </summary>
    /// <param name="input">The <see cref="TensorInfo"/> instance that describes the input shape</param>
    /// <param name="kernel">The size info on the convolutional kernels to apply to the input data</param>
    public delegate ConvolutionInfo ConvolutionInfoFactory(TensorInfo input, (int X, int Y) kernel);
}
