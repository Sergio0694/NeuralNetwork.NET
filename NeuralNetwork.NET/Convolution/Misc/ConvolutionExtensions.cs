using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Convolution.Misc
{
    public static class ConvolutionExtensions
    {
        /// <summary>
        /// Returns the normalized matrix with a max value of 1
        /// </summary>
        /// <param name="m">The input matrix to normalize</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Normalize([NotNull] this float[,] m)
        {
            // Prepare the result matrix
            int h = m.GetLength(0), w = m.GetLength(1);
            float[,] result = new float[h, w];

            // Pool the input matrix
            unsafe
            {
                fixed (float* p = m, r = result)
                {
                    // Get the max value
                    float max = 0;
                    for (int i = 0; i < m.Length; i++)
                        if (p[i] > max) max = p[i];

                    // Normalize the matrix content
                    for (int i = 0; i < m.Length; i++)
                        r[i] = p[i] / max;
                }
            }
            return result;
        }

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="m">The input matrix to pool</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Pool2x2([NotNull] this float[,] m)
        {
            // Prepare the result matrix
            int h = m.GetLength(0), w = m.GetLength(1);
            float[,] result = new float[h / 2 + (h % 2 == 0 ? 0 : 1), w / 2 + (w % 2 == 0 ? 0 : 1)];

            // Pool the input matrix
            int x = 0;
            for (int i = 0; i < h; i += 2)
            {
                int y = 0;
                if (i == h - 1)
                {
                    // Last row
                    for (int j = 0; j < w; j += 2)
                    {
                        float max;
                        if (j == w - 1)
                        {
                            // Last column
                            max = m[i, j];
                        }
                        else max = m[i, j] > m[i, j + 1] ? m[i, j] : m[i, j + 1];
                        result[x, y++] = max;
                    }
                }
                else
                {
                    for (int j = 0; j < w; j += 2)
                    {
                        float max;
                        if (j == w - 1)
                        {
                            // Last column
                            max = m[i, j] > m[i + 1, j] ? m[i, j] : m[i + 1, j];
                        }
                        else
                        {
                            float
                                maxUp = m[i, j] > m[i, j + 1] ? m[i, j] : m[i, j + 1],
                                maxDown = m[i + 1, j] > m[i + 1, j + 1] ? m[i + 1, j] : m[i + 1, j + 1];
                            max = maxUp > maxDown ? maxUp : maxDown;
                        }
                        result[x, y++] = max;
                    }
                }
                x++;
            }
            return result;
        }

        /// <summary>
        /// Performs the Rectified Linear Units operation on the input matrix (applies a minimum value of 0)
        /// </summary>
        /// <param name="m">The input matrix to read</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ReLU([NotNull] this float[,] m)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            float[,] result = new float[h, w];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    result[i, j] = m[i, j] >= 0 ? m[i, j] : 0;
            return result;
        }

        /// <summary>
        /// Convolutes the input matrix with the given 3x3 kernel
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="kernel">The 3x3 convolution kernel to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Convolute3x3([NotNull] this float[,] m, [NotNull] float[,] kernel)
        {
            // Prepare the output matrix
            if (kernel.GetLength(0) != 3 || kernel.GetLength(1) != 3) throw new ArgumentOutOfRangeException("The input kernel must be 3x3");
            int h = m.GetLength(0), w = m.GetLength(1);
            if (h < 3 || w < 3) throw new ArgumentOutOfRangeException("The input matrix must be at least 3x3");
            float[,] result = new float[h - 2, w - 2];

            // Calculate the normalization factor
            float factor = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    factor += kernel[i, j].Abs();

            // Process the convolution
            int x = 0;
            for (int i = 1; i < h - 1; i++)
            {
                int y = 0;
                for (int j = 1; j < w - 1; j++)
                {
                    float
                        partial =
                            m[i - 1, j - 1] * kernel[0, 0] +
                            m[i - 1, j] * kernel[0, 1] +
                            m[i - 1, j + 1] * kernel[0, 2] +
                            m[i, j - 1] * kernel[1, 0] +
                            m[i, j] * kernel[1, 1] +
                            m[i, j + 1] * kernel[1, 2] +
                            m[i + 1, j - 1] * kernel[2, 0] +
                            m[i + 1, j] * kernel[2, 1] +
                            m[i + 1, j + 1] * kernel[2, 2],
                        normalized = partial / factor;
                    result[x, y++] = normalized;
                }
                x++;
            }
            return result;
        }

        /// <summary>
        /// Performs a convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="subdivision">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="mode">The desired convolution mode to use to process the input matrix</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Convolute3x3([NotNull] this float[,] source, int subdivision, [NotNull] IReadOnlyList<float[,]> kernels, ConvolutionMode mode)
        {
            // Checks
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (subdivision < 1) throw new ArgumentOutOfRangeException(nameof(subdivision), "The number of images per row can't be lower than 1");
            if (kernels.Count == 0) throw new ArgumentException(nameof(kernels), "The kernels list can't be empty");
            int
                kh = kernels[0].GetLength(0),
                kw = kernels[0].GetLength(1);
            if (kh != kw) throw new ArgumentException(nameof(kernels), "The kernel must be a square matrix");
            if (kh < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            if (kernels.Any(k => k.GetLength(0) != kh || k.GetLength(1) != kw))
                throw new ArgumentException(nameof(kernels), "One of the input kernels doesn't have a valid size");

            // Local parameters
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                klen = kernels.Count,
                ksize = kh * kw,
                imgSize = w % subdivision == 0 ? w / subdivision : throw new ArgumentException(nameof(source), "Invalid subdivision parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            if (imgSize < ksize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            
            // Valid convolution, result smaller than original
            if (mode == ConvolutionMode.Valid)
            {
                int
                inner = imgAxis - kh + 1,                                       // Size of each image edge after the convolution
                convolutionOutputSize = inner * inner,                          // Size of each processed image
                finalWidth = inner * inner * subdivision * klen,                // Final size of each sample row
                iterationsPerSample = subdivision * klen;                       // Iterations for each dataset entry

                // Process the whole data in a single step
                float[,] result = new float[h, finalWidth];

                // Convolution kernel
                unsafe void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / iterationsPerSample,            // Sample index
                        i_mod = index % iterationsPerSample,
                        j = i_mod / klen,                           // Subdivision index
                        k = i_mod % klen;                           // Kernel index

                    // Read and write sample and subdivision offsets
                    int
                        sourceOffset = i * w + j * imgSize,
                        resultOffset = i * finalWidth + j * convolutionOutputSize;
                    fixed (float* pk = kernels[k], psource = source, presult = result)
                    {
                        for (int x = 0; x < inner; x++)
                        {
                            int offset = resultOffset + x * inner;
                            for (int y = 0; y < inner; y++)
                            {
                                // Process a single convolution of variable size
                                float convolution = 0;
                                for (int kx = 0; kx < kh; kx++)
                                {
                                    int
                                        kOffset = kx * kh,
                                        ksourceOffset = sourceOffset + (kx + x) * imgAxis + y;
                                    for (int ky = 0; ky < kh; ky++)
                                        convolution += pk[kOffset + ky] * psource[ksourceOffset + ky];
                                }
                                presult[offset + y] = convolution;
                            }
                        }
                    }
                }

                // Convolute in parallel
                Parallel.For(0, h * iterationsPerSample, Kernel).AssertCompleted();
                return result;
            }

            // Full convolution
            if (mode == ConvolutionMode.Full)
            {
                throw new NotImplementedException();
            }
            throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported convolution mode");
        }
    }
}
