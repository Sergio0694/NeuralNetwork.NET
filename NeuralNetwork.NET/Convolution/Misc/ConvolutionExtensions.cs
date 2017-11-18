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
        /// Performs a convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceDepth">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsDepth">The depth of each input kernel volume</param>
        /// <param name="mode">The desired convolution mode to use to process the input matrix</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Convolute([NotNull] this float[,] source, int sourceDepth, [NotNull] float[,] kernels, int kernelsDepth, ConvolutionMode mode)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (sourceDepth < 1) throw new ArgumentOutOfRangeException(nameof(sourceDepth), "The number of images per row can't be lower than 1");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            if (kernelsDepth < 1) throw new ArgumentException(nameof(kernelsDepth), "The number of kernels per row must be positive");
            int
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kSize = kw / kernelsDepth,
                kAxis = kSize.IntegerSquare();
            if (kAxis * kAxis != kSize) throw new ArgumentException(nameof(kernels), "The size of the input kernels isn't valid");
            if (kSize < 4) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % sourceDepth == 0 ? w / sourceDepth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");

            /* ============================
             * Valid convolution (forward)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (same depth as the input, each kernel is a 3D volume)
             * Output:          kernelsDepth slices, one for each 3D kernel used */
            if (mode == ConvolutionMode.Forward)
            {
                if (sourceDepth != kernelsDepth) throw new InvalidOperationException("The depth of each kernel must be equal to the depth of each input volume");
                int
                    hResult = imgAxis - kAxis + 1,                      // Size of each image edge after the convolution
                    convolutionOutputSize = hResult * hResult,          // Size of each processed image
                    finalWidth = convolutionOutputSize * nKernels;      // Final size of each sample row

                // Process the valid convolution
                float[,] result = new float[h, finalWidth];
                unsafe void ForwardKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / nKernels,     // Sample index
                        k = index % nKernels;           // Kernel index

                    // Process the current convolution slice
                    int
                        targetBaseOffset = iSample * finalWidth + k * convolutionOutputSize,
                        sourceBaseOffset = iSample * w,
                        kernelBaseOffset = k * kw;
                    fixed (float* psource = source, pkernels = kernels, presult = result)
                    {
                        for (int i = 0; i < hResult; i++)
                        {
                            int
                                targetRowOffset = targetBaseOffset + i * hResult,
                                xEnd = i + kAxis - 1;
                            for (int j = 0; j < hResult; j++)
                            {
                                int highY = j + kAxis - 1;
                                float temp = 0.0f;
                                for (int z = 0; z < sourceDepth; z++)
                                {
                                    int
                                        sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                        kernelDepthOffset = kernelBaseOffset + z * kSize;
                                    for (int x = i; x <= xEnd; x++)
                                    {
                                        int
                                            sourceRowOffset = sourceDepthOffset + x * imgAxis,
                                            kernelRowOffset = kernelDepthOffset + (xEnd - x) * kAxis + highY;
                                        for (int y = j; y <= highY; y++)
                                        {
                                            temp += psource[sourceRowOffset + y] * pkernels[kernelRowOffset - y];
                                        }
                                    }
                                }
                                presult[targetRowOffset + j] = temp;
                            }
                        }
                    }
                }
                Parallel.For(0, h * nKernels, ForwardKernel).AssertCompleted();
                return result;
            }

            /* ============================
             * Full convolution (backwards)
             * ============================
             * Input volume:    H*W*sourceDepth (the delta(l + 1) for each sample)
             * Kernels:         HK*WK*kernelsDepth*sourceDepth (a kernel for each input slice)
             * Output:          kernelsDepth slices, each is the sum of the i-th slice of all the kernelsDepth kernels with convoluted with the i-th input slice */
            if (mode == ConvolutionMode.Backwards)
            {
                int
                    hResult = imgAxis + kAxis - 1,                      // Size of each image edge after the convolution
                    convolutionOutputSize = hResult * hResult,          // Size of each processed image
                    finalWidth = convolutionOutputSize * nKernels;      // Final size of each sample row

                // Process the full convolution
                float[,] result = new float[h, finalWidth];
                unsafe void BackwardsKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / nKernels,     // Sample index
                        iKernel = index % nKernels;     // Kernel index

                    // Process the convolution slice
                    int
                        targetBaseOffset = iSample * finalWidth + iKernel * convolutionOutputSize,
                        sourceBaseOffset = iSample * w,
                        kernelBaseOffset = iKernel * kw;
                    fixed (float* psource = source, pkernels = kernels, presult = result)
                    {
                        for (int i = 0; i < hResult; ++i)
                        {
                            int
                                lowX = 0.Max(i - kAxis + 1),
                                highX = (imgAxis - 1).Min(i),
                                targetRowOffset = targetBaseOffset + i * hResult;
                            for (int j = 0; j < hResult; ++j)
                            {
                                int
                                    lowY = 0.Max(j - kAxis + 1),
                                    highY = (imgAxis - 1).Min(j);
                                float temp = 0.0f;
                                for (int z = 0; z < kernelsDepth; z++)
                                {
                                    int
                                        sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                        kernelDepthOffset = kernelBaseOffset + z * kSize;
                                    for (int x = lowX; x <= highX; ++x)
                                    {
                                        int
                                            sourceRowOffset = sourceDepthOffset + x * imgAxis,
                                            kernelRowOffset = kernelDepthOffset + (i - x) * kAxis + j;
                                        for (int y = lowY; y <= highY; ++y)
                                        {
                                            temp += psource[sourceRowOffset + y] * pkernels[kernelRowOffset - y];
                                        }
                                    }
                                }
                                presult[targetRowOffset + j] = temp;
                            }
                        }
                    }
                }
                Parallel.For(0, h * nKernels, BackwardsKernel).AssertCompleted();
                return result;
            }

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */
            if (mode == ConvolutionMode.Gradient)
            {
                if (nKernels != h) throw new ArgumentException(nameof(kernels), "There must be a delta volume for each activation sample");
                int
                    hResult = imgAxis - kAxis + 1,                              // Size of each image edge after the convolution
                    convolutionOutputSize = hResult * hResult,                  // Size of each processed image
                    gradientSize = convolutionOutputSize * sourceDepth,         // Size of each calculated gradient (one for each original kernel, so for each input delta)
                    finalWidth = gradientSize * kernelsDepth,                   // Final size of each sample row
                    iterationsPerSample = sourceDepth * kernelsDepth;           // Each sample has its own list of 3D gradients, one for each kernel
                    

                // Helper function that calculates the 2D valid convolution between two matrices
                unsafe void Convolute2D(float* psource, float* pkernel, float* presult)
                {
                    for (int i = 0; i < imgAxis; i++)
                    {
                        int
                            targetRowOffset = i * imgAxis,
                            xEnd = i + kAxis - 1;
                        for (int j = 0; j < imgAxis; j++)
                        {
                            int highY = j + kAxis - 1;
                            float temp = 0.0f;
                            for (int x = i; x <= xEnd; x++)
                            {
                                int
                                    sourceRowOffset = x * imgAxis,
                                    kernelRowOffset = (xEnd - x) * kAxis + highY;
                                for (int y = j; y <= highY; y++)
                                {
                                    temp += psource[sourceRowOffset + y] * pkernel[kernelRowOffset - y];
                                }
                            }
                            presult[targetRowOffset + j] = temp;
                        }
                    }
                }

                // Process the valid convolution
                float[,] result = new float[h, finalWidth];
                unsafe void GradientKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / iterationsPerSample,      // Sample index
                        iMod = index % iterationsPerSample,
                        iSampleDepth = iMod / kernelsDepth,         // Depth of the current gradient
                        iKernelDepth = iMod % sourceDepth;          // Output gradient index

                    fixed (float* psource = source, pkernels = kernels, presult = result)
                    {
                        // Adjust the output offsets
                        float*
                            psourceTarget = psource + iSample * w + iSampleDepth * imgSize,
                            pkernelsTarget = pkernels + iKernelDepth * kSize,
                            presultTarget = presult + iSample * finalWidth + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;
                        Convolute2D(psourceTarget, pkernelsTarget, presultTarget);
                    }
                }
                Parallel.For(0, h * iterationsPerSample, GradientKernel).AssertCompleted();
                return result;
            }
            throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported convolution mode");
        }
    }
}
