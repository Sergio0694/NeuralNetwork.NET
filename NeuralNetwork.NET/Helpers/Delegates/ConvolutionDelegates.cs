using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Helpers.Delegates
{
    /// <summary>
    /// A delegates that wraps a method that performs a forward convolution operation on the source matrix, using the given kernels
    /// </summary>
    /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
    /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
    /// <param name="kernels">The list of convolution kernels to apply to each image</param>
    /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
    /// <param name="biases">The bias vector to sum to the resulting images</param>
    /// <param name="result">The resulting convolution volume</param>
    /// <exception cref="System.ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
    [NotNull]
    public delegate void ForwardConvolution(
        in FloatSpan2D source, in VolumeInformation sourceInfo,
        [NotNull] float[,] kernels, in VolumeInformation kernelsInfo,
        [NotNull] float[] biases,
        out FloatSpan2D result);

    /// <summary>
    /// A delegates that wraps a method that performs a full backwards convolution operation on the source matrix, using the given kernels
    /// </summary>
    /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
    /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
    /// <param name="kernels">The list of convolution kernels to apply to each image</param>
    /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
    /// <param name="result">The resulting convolution volume</param>
    /// <exception cref="System.ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
    [NotNull]
    public delegate void BackwardsConvolution(
        in FloatSpan2D source, in VolumeInformation sourceInfo,
        in FloatSpan2D kernels, in VolumeInformation kernelsInfo,
        out FloatSpan2D result);

    /// <summary>
    /// A delegates that wraps a method that performs a the gradient convolution operation on the source matrix, using the given kernels
    /// </summary>
    /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
    /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
    /// <param name="kernels">The list of convolution kernels to apply to each image</param>
    /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
    /// <param name="result">The resulting convolution volume</param>
    /// <exception cref="System.ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
    public delegate float[,] GradientConvolution(
        in FloatSpan2D source, in VolumeInformation sourceInfo,
        in FloatSpan2D kernels, in VolumeInformation kernelsInfo,
        out FloatSpan2D result);

}
