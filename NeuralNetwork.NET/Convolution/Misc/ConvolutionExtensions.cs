using System;
using System.Diagnostics;
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
        /// <param name="depth">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="mode">The desired convolution mode to use to process the input matrix</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static unsafe float[,] Convolute([NotNull] this float[,] source, int depth, [NotNull] float[,] kernels, ConvolutionMode mode)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            int
                klen = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                ksize = kw / depth,
                kAxis = ksize.IntegerSquare();
            if (kAxis * kAxis != ksize) throw new ArgumentException(nameof(kernels), "The size of the input kernels isn't valid");
            if (ksize < 4) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            if (imgSize < ksize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            
            // Valid convolution, result smaller than original
            if (mode == ConvolutionMode.Valid)
            {
                int
                    inner = imgAxis - kAxis + 1,                    // Size of each image edge after the convolution
                    convolutionOutputSize = inner * inner,          // Size of each processed image
                    finalWidth = convolutionOutputSize * klen;      // Final size of each sample row

                // Process the whole data in a single step
                float[,] result = new float[h, finalWidth];

                // Convolution kernel
                unsafe void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / klen,   // Sample index
                        k = index % klen;   // Kernel index

                    fixed (float* presult = result, psource = source, pk = kernels)
                    {
                        int
                            sampleOffset = i * w,
                            kOffset = k * ksize,
                            resultOffset = i * finalWidth + k * convolutionOutputSize;
                        for (int x = 0; x < inner; x++)
                        {
                            int
                                hResultOffset = resultOffset + x * inner,
                                xAxisOffset = x * imgAxis;
                            for (int y = 0; y < inner; y++)
                            {
                                float convolution = 0;
                                int xyAxisOffset = xAxisOffset + y;
                                for (int kz = 0; kz < depth; kz++)
                                {
                                    int
                                        kzOffset = kOffset + kz * ksize,
                                        kzSourceOffset = sampleOffset + kz * imgSize + xyAxisOffset;
                                    for (int kx = 0; kx < kAxis; kx++)
                                    {
                                        int
                                            kxOffset = kzOffset + kx * kAxis,
                                            sourceOffset = kzSourceOffset + kx * imgAxis;
                                        for (int ky = 0; ky < kAxis; ky++)
                                            convolution += pk[kxOffset + ky] * psource[sourceOffset + ky];
                                    }
                                }
                                presult[hResultOffset + y] = convolution;
                            }
                        }
                    }
                }

                // Convolute in parallel
                Parallel.For(0, h * klen, Kernel).AssertCompleted();
                return result;
            }

            // Full convolution
            if (mode == ConvolutionMode.Full)
            {
                int
                    edge = kAxis - 1,
                    inner = imgAxis - kAxis + 1 + 2 * edge,         // Size of each image edge after the convolution
                    convolutionOutputSize = inner * inner,          // Size of each processed image
                    finalWidth = convolutionOutputSize * klen;      // Final size of each sample row

                // Process the whole data in a single step
                float[,] result = new float[h, finalWidth];

                for (int index = 0; index < h * klen; index++)
                {
                    // Calculate the current indexes
                    int
                        i = index / klen,   // Sample index
                        k = index % klen;   // Kernel index

                    fixed (float* presult = result, psource = source, pk = kernels)
                    {
                        int
                            sampleOffset = i * w,
                            kOffset = k * ksize,
                            resultOffset = i * finalWidth + k * convolutionOutputSize;
                        for (int x = 0; x < inner; x++)
                        {
                            int
                                hResultOffset = resultOffset + x * inner,
                                xAxisOffset = x * imgAxis;
                            for (int y = 0; y < inner; y++)
                            {
                                float convolution = 0;
                                int xyAxisOffset = xAxisOffset + y;
                                for (int kz = 0; kz < depth; kz++)
                                {
                                    int
                                        kzOffset = kOffset + kz * ksize,
                                        kzSourceOffset = sampleOffset + kz * imgSize + xyAxisOffset;

                                    int kxStart, kxEnd;
                                    if (x >= inner + edge)
                                    {
                                        kxStart = 0;
                                        kxEnd = inner - x;
                                    }
                                    else if (x < edge)
                                    {
                                        kxStart = kAxis - x - 1;
                                        kxEnd = kAxis;
                                    }
                                    else
                                    {
                                        kxStart = 0;
                                        kxEnd = kAxis;
                                    }
                                    Debug.Assert(0.Max(kAxis - x - 1) == kxStart);
                                    Debug.Assert(kAxis.Min(inner - x + 1) == kxEnd);

                                    for (int kx = kxStart; kx < kxEnd; kx++)
                                    {
                                        int
                                            kxOffset = kzOffset + kx * kAxis,
                                            sourceOffset = kzSourceOffset + (imgAxis - kx - 1) * imgAxis;
                                        int sourceX = imgAxis - kx - 1;

                                        int kyStart, kyEnd, yStart;
                                        if (y >= inner + edge)
                                        {
                                            yStart = imgAxis - kAxis + 1;
                                            kyStart = 0;
                                            kyEnd = inner - y;
                                        }
                                        else if (y < edge)
                                        {
                                            yStart = 0;
                                            kyStart = kAxis - y - 1;
                                            kyEnd = kAxis;
                                        }
                                        else
                                        {
                                            kyStart = yStart = 0;
                                            kyEnd = kAxis;
                                        }
                                        for (int ky = kyStart; ky < kyEnd; ky++)
                                        {
                                            int sourceY = yStart + ky;





                                            float pkv = pk[kxOffset + ky];
                                            float psv = psource[sourceOffset + sourceY];
                                            
                                            convolution += pk[kxOffset + ky] * psource[sourceOffset + imgAxis - ky - 1];
                                        }
                                    }
                                }
                                presult[hResultOffset + y] = convolution;
                            }
                        }
                    }
                }

                // Convolute in parallel
                //Parallel.For(0, h * klen, Kernel).AssertCompleted();
                return result;
            }
            throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported convolution mode");
        }
    }
}
