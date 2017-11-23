using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Helpers
{
    public static class ConvolutionExtensions
    {
        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="source">The input matrix to pool</param>
        /// <param name="depth">The number of images for each matrix row</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Pool2x2([NotNull] this float[,] source, int depth)
        {
            // Prepare the result matrix
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per sample must be at least equal to 1");
            int h = source.GetLength(0), w = source.GetLength(1);
            if (h < 1 || w < 1) throw new ArgumentException("The input matrix isn't valid");
            int
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            float[,] result = new float[h, poolFinalWidth];

            // Pooling kernel
            unsafe void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * w,
                    resultBaseOffset = sample * poolFinalWidth;
                fixed (float* psource = source, presult = result)
                {
                    for (int z = 0; z < depth; z++)
                    {
                        int
                            sourceZOffset = sourceBaseOffset + z * imgSize,
                            resultZOffset = resultBaseOffset + z * poolSize,
                            x = 0;
                        for (int i = 0; i < imgAxis; i += 2)
                        {
                            int
                                sourceIOffset = sourceZOffset + i * imgAxis,
                                resultXOffset = resultZOffset + x * poolAxis,
                                y = 0;
                            if (i == edge)
                            {
                                // Last row
                                for (int j = 0; j < imgAxis; j += 2)
                                {
                                    float max;
                                    if (j == w - 1) max = psource[sourceIOffset + j]; // Last column
                                    else
                                    {
                                        float
                                            left = psource[sourceIOffset + j],
                                            right = psource[sourceIOffset + j + 1];
                                        max = left > right ? left : right;
                                    }
                                    presult[resultXOffset + y++] = max;
                                }
                            }
                            else
                            {
                                int sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                                for (int j = 0; j < imgAxis; j += 2)
                                {
                                    float max;
                                    if (j == edge)
                                    {
                                        // Last column
                                        float
                                            up = psource[sourceIOffset + j],
                                            down = psource[sourceI_1Offset + j];
                                        max = up > down ? up : down;
                                    }
                                    else
                                    {
                                        float
                                            upLeft = psource[sourceIOffset + j],
                                            upRight = psource[sourceIOffset + j + 1],
                                            downLeft = psource[sourceI_1Offset + j],
                                            downRight = psource[sourceI_1Offset + j + 1],
                                            maxUp = upLeft > upRight ? upLeft : upRight,
                                            maxDown = downLeft > downRight ? downLeft : downRight;
                                        max = maxUp > maxDown ? maxUp : maxDown;
                                    }
                                    presult[resultXOffset + y++] = max;
                                }
                            }
                            x++;
                        }
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="source">The activation matrix</param>
        /// <param name="pooled">The matrix to upscale according to the source values</param>
        /// <param name="depth">The number of images for each matrix row</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] UpscalePool2x2([NotNull] this float[,] source, [NotNull] float[,] pooled, int depth)
        {
            // Prepare the result matrix
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per sample must be at least equal to 1");
            int h = source.GetLength(0), w = source.GetLength(1);
            if (h < 1 || w < 1) throw new ArgumentException("The input matrix isn't valid");
            int
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            int
                poolAxis = imgAxis / 2 + (imgAxis % 2 == 0 ? 0 : 1),
                poolSize = poolAxis * poolAxis,
                poolFinalWidth = depth * poolSize,
                edge = imgAxis - 1;
            int
                ph = pooled.GetLength(0),
                pw = pooled.GetLength(1);
            if (ph != h || pw != poolFinalWidth) throw new ArgumentException("Invalid pooled matrix", nameof(pooled));
            float[,] result = new float[h, w];

            // Pooling kernel
            unsafe void Kernel(int sample)
            {
                int
                    sourceBaseOffset = sample * w,
                    resultBaseOffset = sample * poolFinalWidth;
                fixed (float* psource = source, ppooled = pooled, presult = result)
                {
                    for (int z = 0; z < depth; z++)
                    {
                        int
                            sourceZOffset = sourceBaseOffset + z * imgSize,
                            resultZOffset = resultBaseOffset + z * poolSize,
                            x = 0;
                        for (int i = 0; i < imgAxis; i += 2)
                        {
                            int
                                sourceIOffset = sourceZOffset + i * imgAxis,
                                resultXOffset = resultZOffset + x * poolAxis,
                                y = 0;
                            if (i == edge)
                            {
                                // Last row
                                for (int j = 0; j < imgAxis; j += 2)
                                {
                                    if (j == w - 1)
                                    {
                                        presult[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                    }
                                    else
                                    {
                                        float
                                            left = psource[sourceIOffset + j],
                                            right = psource[sourceIOffset + j + 1];
                                        if (left > right) presult[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                        else presult[sourceIOffset + j + 1] = ppooled[resultXOffset + y++];
                                    }
                                }
                            }
                            else
                            {
                                int sourceI_1Offset = sourceZOffset + (i + 1) * imgAxis;
                                for (int j = 0; j < imgAxis; j += 2)
                                {
                                    if (j == edge)
                                    {
                                        // Last column
                                        float
                                            up = psource[sourceIOffset + j],
                                            down = psource[sourceI_1Offset + j];
                                        if (up > down) presult[sourceIOffset + j] = ppooled[resultXOffset + y++];
                                        else presult[sourceI_1Offset + j] = ppooled[resultXOffset + y++];
                                    }
                                    else
                                    {
                                        int offset = sourceIOffset + j;
                                        float
                                            max = psource[offset],
                                            next = psource[sourceIOffset + j + 1];
                                        if (next > max)
                                        {
                                            max = next;
                                            offset = sourceIOffset + j + 1;
                                        }
                                        next = psource[sourceI_1Offset + j];
                                        if (next > max)
                                        {
                                            max = next;
                                            offset = sourceI_1Offset + j;
                                        }
                                        next = psource[sourceI_1Offset + j + 1];
                                        if (next > max)
                                        {
                                            offset = sourceI_1Offset + j + 1;
                                        }
                                        presult[offset] = ppooled[resultXOffset + y++];
                                    }
                                }
                            }
                            x++;
                        }
                    }
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Compresses a convolution matrix into a row vector by summing each 2D slice in each row
        /// </summary>
        /// <param name="source">The matrix to compress</param>
        /// <param name="depth">The number of images per row</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[] CompressVertically([NotNull] this float[,] source, int depth)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            float[,] result = new float[h, depth];

            // Kernel to sum each slice
            unsafe void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / depth,    // Sample index
                    z = index % depth;          // 2D slice index

                // Reverse the input matrix sequentially
                int baseOffset = iSample * w + z * imgSize;
                fixed (float* presult = result, psource = source)
                {
                    float sum = 0;
                    for (int i = 0; i < imgSize; i++)
                    {
                        sum += psource[baseOffset + i];
                    }
                    presult[iSample * depth + z] = sum;
                }
            }
            Parallel.For(0, h * depth, Kernel).AssertCompleted();
            return result.CompressVertically();
        }

        /// <summary>
        /// Rotates the input volume by 180 degrees
        /// </summary>
        /// <param name="source">The input matrix to rotate</param>
        /// <param name="depth">The number of images per row</param>
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Rotate180([NotNull] this float[,] source, int depth)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            int
                threshold = imgSize / 2,
                edge = imgSize - 1;
            bool odd = imgSize % 2 == 1;
            float[,] result = new float[h, w];

            // Inversion kernel
            unsafe void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / depth,    // Sample index
                    z = index % depth;          // 2D slice index

                // Reverse the input matrix sequentially
                int baseOffset = iSample * w + z * imgSize;
                fixed (float* presult = result, psource = source)
                {
                    for (int i = 0; i < threshold; i++)
                    {
                        int
                            left = baseOffset + i,
                            right = baseOffset + edge - i;
                        presult[left] = psource[right];
                        presult[right] = psource[left];
                    }
                    if (odd)
                    {
                        int center = baseOffset + threshold;
                        presult[center] = psource[center];
                    }
                }
            }
            Parallel.For(0, h * depth, Kernel).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Sums a vector to each 2D slice in the input data volume
        /// </summary>
        /// <param name="source">The input volume, where each row is a separate 3D volume of the given depth</param>
        /// <param name="depth">The number of images in each row</param>
        /// <param name="v">The vector to sum</param>
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void InPlaceSum([NotNull] this float[,] source, int depth, [NotNull] float[] v)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth), "The number of images per row can't be lower than 1");
            if (v.Length == 0) throw new ArgumentException(nameof(v), "The sum vector can't be empty");
            if (v.Length != depth) throw new ArgumentException("The sum vector must be as long as the depth of the input volume");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % depth == 0 ? w / depth : throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");

            // Add the bias to each 2D slice
            unsafe void Kernel(int i)
            {
                int baseOffset = i * w;
                fixed (float* psource = source, pv = v)
                {
                    for (int z = 0; z < depth; z++)
                    {
                        int sliceOffset = baseOffset + z * imgSize;
                        float sum = pv[z];
                        for (int j = 0; j < imgSize; j++)
                            psource[sliceOffset + j] += sum;
                    }
                }
            }
            Parallel.For(0, h, Kernel);
        }

        /// <summary>
        /// Performs a forward convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceDepth">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsDepth">The depth of each input kernel volume</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteForward([NotNull] this float[,] source, int sourceDepth, [NotNull] float[,] kernels, int kernelsDepth)
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
            if (sourceDepth != kernelsDepth) throw new InvalidOperationException("The depth of each kernel must be equal to the depth of each input volume");

            /* ============================
             * Valid convolution (forward)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (same depth as the input, each kernel is a 3D volume)
             * Output:          kernelsDepth slices, one for each 3D kernel used */
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

        /// <summary>
        /// Performs the full backwards convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceDepth">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsDepth">The depth of each input kernel volume</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteBackwards([NotNull] this float[,] source, int sourceDepth, [NotNull] float[,] kernels, int kernelsDepth)
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
            if (sourceDepth != nKernels) throw new ArgumentException("The source depth must be equal to the number of kernels");

            /* ============================
             * Full convolution (backwards)
             * ============================
             * Input volume:    H*W*sourceDepth (the delta(l + 1) for each sample)
             * Kernels:         HK*WK*kernelsDepth*sourceDepth (a kernel for each input slice)
             * Output:          kernelsDepth slices, each is the sum of the i-th slice of all the kernelsDepth kernels with convoluted with the i-th input slice */
            int
                hResult = imgAxis + kAxis - 1,                          // Size of each image edge after the convolution
                convolutionOutputSize = hResult * hResult,              // Size of each processed image
                finalWidth = convolutionOutputSize * kernelsDepth;      // Final size of each sample row

            // Process the full convolution
            float[,] result = new float[h, finalWidth];
            unsafe void BackwardsKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / kernelsDepth,         // Sample index
                    iKernelDepth = index % kernelsDepth;    // Kernel index

                // Process the convolution slice
                int
                    targetBaseOffset = iSample * finalWidth + iKernelDepth * convolutionOutputSize,
                    sourceBaseOffset = iSample * w,
                    kernelBaseOffset = iKernelDepth * kSize;
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
                            for (int z = 0; z < nKernels; z++)
                            {
                                int
                                    sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                    kernelDepthOffset = kernelBaseOffset + z * kw;
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
            Parallel.For(0, h * kernelsDepth, BackwardsKernel).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Performs a the gradient convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceDepth">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsDepth">The depth of each input kernel volume</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteGradient([NotNull] this float[,] source, int sourceDepth, [NotNull] float[,] kernels, int kernelsDepth)
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
            if (nKernels != h) throw new ArgumentException(nameof(kernels), "There must be a delta volume for each activation sample");

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */
            int
                hResult = imgAxis - kAxis + 1,                              // Size of each image edge after the convolution
                convolutionOutputSize = hResult * hResult,                  // Size of each processed image
                gradientSize = convolutionOutputSize * sourceDepth,         // Size of each calculated gradient (one for each original kernel, so for each input delta)
                finalWidth = gradientSize * kernelsDepth,                   // Final size of each sample row
                iterationsPerSample = sourceDepth * kernelsDepth;           // Each sample has its own list of 3D gradients, one for each kernel

            // Process the valid convolution
            float[,] result = new float[h, finalWidth];
            unsafe void GradientKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / iterationsPerSample,      // Sample index
                    iMod = index % iterationsPerSample,
                    iSampleDepth = iMod / kernelsDepth,         // Depth of the current gradient
                    iKernelDepth = iMod % kernelsDepth;         // Output gradient index

                // Process the current convolution slice
                int
                    sourceBaseOffset = iSample * w + iSampleDepth * imgSize,
                    kernelBaseOffset = iSample * kw + iKernelDepth * kSize,
                    resultBaseOffset = iSample * finalWidth + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;
                fixed (float* psource = source, pkernels = kernels, presult = result)
                {
                    for (int i = 0; i < hResult; i++)
                    {
                        int
                            targetRowOffset = resultBaseOffset + i * hResult,
                            xEnd = i + kAxis - 1;
                        for (int j = 0; j < hResult; j++)
                        {
                            int highY = j + kAxis - 1;
                            float temp = 0.0f;
                            for (int x = i; x <= xEnd; x++)
                            {
                                int
                                    sourceRowOffset = sourceBaseOffset + x * imgAxis,
                                    kernelRowOffset = kernelBaseOffset + (xEnd - x) * kAxis + highY;
                                for (int y = j; y <= highY; y++)
                                {
                                    temp += psource[sourceRowOffset + y] * pkernels[kernelRowOffset - y];
                                }
                            }
                            presult[targetRowOffset + j] = temp;
                        }
                    }

                }
            }
            Parallel.For(0, h * iterationsPerSample, GradientKernel).AssertCompleted();
            return result;
        }
    }
}
