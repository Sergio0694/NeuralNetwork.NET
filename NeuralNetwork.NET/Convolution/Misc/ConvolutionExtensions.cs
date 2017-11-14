using System;
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
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kSize = kw / depth,
                kAxis = kSize.IntegerSquare();
            if (kAxis * kAxis != kSize) throw new ArgumentException(nameof(kernels), "The size of the input kernels isn't valid");
            if (kSize < 4) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            
            // Valid convolution, result smaller than original
            if (mode == ConvolutionMode.Valid)
            {
                int
                    hResult = imgAxis - kAxis + 1,                      // Size of each image edge after the convolution
                    convolutionOutputSize = hResult * hResult,          // Size of each processed image
                    finalWidth = convolutionOutputSize * nKernels;      // Final size of each sample row

                // Process the whole data in a single step
                float[,] result = new float[h, finalWidth];

                fixed (float* src = source, kernel = kernels, dst = result)
                {
                    for (int iSample = 0; iSample < h; iSample++)
                    {
                        for (int iKernel = 0; iKernel < nKernels; iKernel++)
                        {
                            for (int i = 0; i < hResult; ++i)
                            {
                                for (int j = 0; j < hResult; ++j)
                                {
                                    float temp = 0.0f;
                                    for (int z = 0; z < depth; z++)
                                    {
                                        for (int k = i; k <= i + kAxis - 1; ++k)
                                        {
                                            for (int l = j; l <= j + kAxis - 1; ++l)
                                            {
                                                temp += 
                                                    src[iSample * w + z * imgSize + k * imgAxis + l] * 
                                                    kernel[iKernel * kw + z * kSize + (i + kAxis - 1 - k) * kAxis + (j + kAxis - 1 - l)];
                                            }
                                        }
                                    }
                                    dst[iSample * finalWidth + iKernel * convolutionOutputSize + i * hResult + j] = temp;
                                }
                            }
                        }
                    }
                }

                return result;

                unsafe void ValidKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / nKernels,   // Sample index
                        k = index % nKernels;   // Kernel index

                    fixed (float* psource = source, pk = kernels, presult = result)
                    {
                        for (int x = 0; x < hResult; x++)
                        {
                            int 
                                lowK = 0.Max(x - kAxis + 1),
                                highK = (imgAxis - 1).Min(x);
                            for (int y = 0; y < hResult; y++)
                            {
                                int 
                                    lowL = 0.Max(y - kAxis + 1),
                                    highL = (imgAxis - 1).Min(y);
                                float temp = 0f;
                                for (int z = 0; z < depth; z++)
                                {
                                    for (k = lowK; k <= highK; ++k)
                                    {
                                        for (int l = lowL; l <= highL; ++l)
                                        {
                                            temp += psource[i * w + z * imgSize + k * imgAxis + l] * pk[k * kw + z * kSize + (x - k) * kAxis + (y - l)];
                                        }
                                    }
                                }
                                presult[i * finalWidth + k * convolutionOutputSize + x * hResult + y] = temp;
                            }
                        }
                    }
                }
                Parallel.For(0, h * nKernels, ValidKernel).AssertCompleted();
                return result;
            }

            // Full convolution
            if (mode == ConvolutionMode.Full)
            {
                int
                    hResult = imgAxis + kAxis - 1,                      // Size of each image edge after the convolution
                    convolutionOutputSize = hResult * hResult,          // Size of each processed image
                    finalWidth = convolutionOutputSize * nKernels;      // Final size of each sample row

                
            }
            throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported convolution mode");
        }

        // Backup 2D method
        public static unsafe float[] convolute(float[] mSource, int h_src, int w_src, float[] mKernel, int h_kernel, int w_kernel, ConvolutionMode mode)
        {
            int h_dst, w_dst;
            switch (mode)
            {
                case ConvolutionMode.Full:
                    h_dst = h_src + h_kernel - 1;
                    w_dst = w_src + w_kernel - 1;
                    break;
                case ConvolutionMode.Valid:
                    h_dst = h_src - h_kernel + 1;
                    w_dst = w_src - w_kernel + 1;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(mode));
            }

            if (h_dst <= 0 || w_dst <= 0) throw new InvalidOperationException();

            float[] dst = new float[h_dst * w_dst];



            fixed (float* src = mSource, kernel = mKernel)
            {
                float temp;
                int i, j, k, l;
                int low_k, high_k, low_l, high_l;
                int i_src, j_src;

                switch (mode)
                {
                    case ConvolutionMode.Full:
                        // Full linear convolution of size N + M -1
                        for (i = 0; i < h_dst; ++i)
                        {
                            low_k = 0.Max(i - h_kernel + 1);
                            high_k = (h_src - 1).Min(i);
                            for (j = 0; j < w_dst; ++j)
                            {
                                low_l = 0.Max(j - w_kernel + 1);
                                high_l = (w_src - 1).Min(j);
                                temp = 0.0f;
                                for (k = low_k; k <= high_k; ++k)
                                {
                                    for (l = low_l; l <= high_l; ++l)
                                    {
                                        temp += src[k * w_src + l] * kernel[(i - k) * w_kernel + (j - l)];
                                    }
                                }
                                dst[i * w_dst + j] = temp;
                            }
                        }
                        break;
                    case ConvolutionMode.Valid:
                        // Valid linear convolution, of size N - M
                        for (i = 0; i < h_dst; ++i)
                        {
                            for (j = 0; j < w_dst; ++j)
                            {
                                temp = 0.0f;
                                for (k = i; k <= i + h_kernel - 1; ++k)
                                {
                                    for (l = j; l <= j + w_kernel - 1; ++l)
                                    {
                                        temp += src[k * w_src + l] * kernel[(i + h_kernel - 1 - k) * w_kernel + (j + w_kernel - 1 - l)];
                                    }
                                }
                                dst[i * w_dst + j] = temp;
                            }
                        }
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return dst;
        }
    }
}
