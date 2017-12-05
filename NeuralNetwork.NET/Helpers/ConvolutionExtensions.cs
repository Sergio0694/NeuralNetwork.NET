﻿using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;

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
        /// Performs a forward convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="biases">The bias vector to sum to the resulting images</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteForward(
            [NotNull] this float[,] source, VolumeInformation sourceInfo,
            [NotNull] float[,] kernels, VolumeInformation kernelsInfo,
            [NotNull] float[] biases)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            int
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                sourceDepth = sourceInfo.Depth,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;  // Size of an edge of one of the inner images per sample
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (sourceInfo.Depth != kernelsInfo.Depth) throw new InvalidOperationException("The depth of each kernel must be equal to the depth of each input volume");
            if (biases.Length == 0) throw new ArgumentException(nameof(biases), "The sum vector can't be empty");
            if (biases.Length != nKernels) throw new ArgumentException("The sum vector must be as long as the depth of the input volume");

            /* ============================
             * Valid convolution (forward)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (same depth as the input, each kernel is a 3D volume)
             * Output:          kernelsDepth slices, one for each 3D kernel used */
            int
                hResult = imgHeight - kHeight + 1,                  // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,          // Size of each processed image
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
                fixed (float* psource = source, pkernels = kernels, pbiases = biases, presult = result)
                {
                    for (int i = 0; i < hResult; i++)
                    {
                        int
                            targetRowOffset = targetBaseOffset + i * hResult,
                            xEnd = i + kHeight - 1;
                        for (int j = 0; j < wResult; j++)
                        {
                            int highY = j + kWidth - 1;
                            float temp = 0.0f;
                            for (int z = 0; z < sourceDepth; z++)
                            {
                                int
                                    sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                    kernelDepthOffset = kernelBaseOffset + z * kSize;
                                for (int x = i; x <= xEnd; x++)
                                {
                                    int
                                        sourceRowOffset = sourceDepthOffset + x * imgWidth,
                                        kernelRowOffset = kernelDepthOffset + (xEnd - x) * kWidth + highY;
                                    for (int y = j; y <= highY; y++)
                                    {
                                        temp += psource[sourceRowOffset + y] * pkernels[kernelRowOffset - y];
                                    }
                                }
                            }
                            presult[targetRowOffset + j] = temp + pbiases[k];
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
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteBackwards([NotNull] this float[,] source, VolumeInformation sourceInfo, [NotNull] float[,] kernels, VolumeInformation kernelsInfo)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            int
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width,
                kDepth = kernelsInfo.Depth;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (sourceInfo.Depth != nKernels) throw new ArgumentException("The source depth must be equal to the number of kernels");

            /* ============================
             * Full convolution (backwards)
             * ============================
             * Input volume:    H*W*sourceDepth (the delta(l + 1) for each sample)
             * Kernels:         HK*WK*kernelsDepth*sourceDepth (a kernel for each input slice)
             * Output:          kernelsDepth slices, each is the sum of the i-th slice of all the kernelsDepth kernels with convoluted with the i-th input slice */
            int
                hResult = imgHeight + kHeight - 1,                  // Size of each image edge after the convolution
                wResult = imgWidth + kWidth - 1,
                convolutionOutputSize = hResult * wResult,          // Size of each processed image
                finalWidth = convolutionOutputSize * kDepth;        // Final size of each sample row

            // Process the full convolution
            float[,] result = new float[h, finalWidth];
            unsafe void BackwardsKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / kDepth,         // Sample index
                    iKernelDepth = index % kDepth;    // Kernel index

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
                            lowX = 0.Max(i - kHeight + 1),
                            highX = (imgHeight - 1).Min(i),
                            targetRowOffset = targetBaseOffset + i * hResult;
                        for (int j = 0; j < hResult; ++j)
                        {
                            int
                                lowY = 0.Max(j - kWidth + 1),
                                highY = (imgWidth - 1).Min(j);
                            float temp = 0.0f;
                            for (int z = 0; z < nKernels; z++)
                            {
                                int
                                    sourceDepthOffset = sourceBaseOffset + z * imgSize,
                                    kernelDepthOffset = kernelBaseOffset + z * kw;
                                for (int x = lowX; x <= highX; ++x)
                                {
                                    int
                                        sourceRowOffset = sourceDepthOffset + x * imgWidth,
                                        kernelRowOffset = kernelDepthOffset + (i - x) * kWidth + j;
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
            Parallel.For(0, h * kDepth, BackwardsKernel).AssertCompleted();
            return result;
        }

        /// <summary>
        /// Performs a the gradient convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] ConvoluteGradient([NotNull] this float[,] source, VolumeInformation sourceInfo, [NotNull] float[,] kernels, VolumeInformation kernelsInfo)
        {
            // Checks and local parameters
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            int
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kDepth = kernelsInfo.Depth,
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (nKernels != h) throw new ArgumentException(nameof(kernels), "There must be a delta volume for each activation sample");

            /* ============================
             * Valid convolution (gradient)
             * ============================
             * Input volume:    H*W*sourceDepth (for each sample)
             * Kernels:         HK*WK*sourceDepth*kernelsDepth (delta(l + 1) used to calculate the 3D gradient for each kernel)
             * Output:          sourceDepth*kernelsDepth slices, where each stack of sourceDepth slices is the gradient for the i-th kernel */
            int
                hResult = imgHeight - kHeight + 1,                          // Size of each image edge after the convolution
                wResult = imgWidth - kWidth + 1,
                convolutionOutputSize = hResult * wResult,                  // Size of each processed image
                gradientSize = convolutionOutputSize * sourceInfo.Depth,    // Size of each calculated gradient (one for each original kernel, so for each input delta)
                finalWidth = gradientSize * kernelsInfo.Depth,              // Final size of each sample row
                iterationsPerSample = sourceInfo.Depth * kDepth;            // Each sample has its own list of 3D gradients, one for each kernel

            // Process the valid convolution
            float[,] result = new float[h, finalWidth];
            unsafe void GradientKernel(int index)
            {
                // Calculate the current indexes
                int
                    iSample = index / iterationsPerSample,      // Sample index
                    iMod = index % iterationsPerSample,
                    iSampleDepth = iMod / kDepth,               // Depth of the current gradient
                    iKernelDepth = iMod % kDepth;               // Output gradient index

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
                            xEnd = i + kHeight - 1;
                        for (int j = 0; j < hResult; j++)
                        {
                            int highY = j + kWidth - 1;
                            float temp = 0.0f;
                            for (int x = i; x <= xEnd; x++)
                            {
                                int
                                    sourceRowOffset = sourceBaseOffset + x * imgWidth,
                                    kernelRowOffset = kernelBaseOffset + (xEnd - x) * kWidth + highY;
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