using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// A static class that contains some GPU-accelerated convolution extension methods
    /// </summary>
    public static class ConvolutionGpuExtensions
    {
        /// <summary>
        /// Performs a forward convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="biases">The bias vector to sum to the resulting images</param>
        /// <param name="result">The resulting convolution matrix</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static void ConvoluteForward(
            in FloatSpan2D source, in VolumeInformation sourceInfo,
            [NotNull] float[,] kernels, in VolumeInformation kernelsInfo,
            [NotNull] float[] biases,
            out FloatSpan2D result)
        {
            // Checks and local parameters
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels can't be empty");
            int
                nKernels = kernels.GetLength(0),
                kw = kernels.GetLength(1),
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.Height,
                w = source.Width,
                sourceDepth = sourceInfo.Depth,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;  // Size of an edge of one of the inner images per sample
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (sourceInfo.Depth != kernelsInfo.Depth) throw new InvalidOperationException("The depth of each kernel must be equal to the depth of each input volume");
            if (biases.Length == 0) throw new ArgumentException(nameof(biases), "The sum vector can't be empty");
            if (biases.Length != nKernels) throw new ArgumentException("The sum vector must be as long as the depth of the input volume");
            Gpu gpu = Gpu.Default;

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
            using (DeviceMemory<float> biases_gpu = gpu.AllocateDevice(biases))
            using (DeviceMemory2D<float>
                result_gpu = gpu.AllocateDevice<float>(h, finalWidth),
                source_gpu = gpu.AllocateDevice(source),
                kernels_gpu = gpu.AllocateDevice(kernels))
            {
                deviceptr<float>
                    presult_gpu = result_gpu.Ptr,
                    psource_gpu = source_gpu.Ptr,
                    pkernels_gpu = kernels_gpu.Ptr,
                    pbiases_gpu = biases_gpu.Ptr;
                int
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32(),
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    kernels_gpu_pitch = kernels_gpu.PitchInElements.ToInt32();

                // Forward GPU kernel
                void ForwardKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / nKernels,     // Sample index
                        k = index % nKernels;           // Kernel index

                    // Process the current convolution slice
                    int
                        targetBaseOffset = iSample * result_gpu_pitch + k * convolutionOutputSize,
                        sourceBaseOffset = iSample * source_gpu_pitch,
                        kernelBaseOffset = k * kernels_gpu_pitch;
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
                                        temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                    }
                                }
                            }
                            presult_gpu[targetRowOffset + j] = temp + pbiases_gpu[k];
                        }
                    }
                }
                gpu.For(0, h * nKernels, ForwardKernel);
                result_gpu.CopyToHost(out result);
            }
        }

        /// <summary>
        /// Performs the full backwards convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="result">The resulting matrix where each row contains the result of the convolutions for each original image for each sample</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static void ConvoluteBackwards(
            in FloatSpan2D source, in VolumeInformation sourceInfo,
            in FloatSpan2D kernels, in VolumeInformation kernelsInfo,
            out FloatSpan2D result)
        {
            // Checks and local parameters
            int
                nKernels = kernels.Height,
                kw = kernels.Width,
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width,
                kDepth = kernelsInfo.Depth;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.Height,
                w = source.Width,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (sourceInfo.Depth != nKernels) throw new ArgumentException("The source depth must be equal to the number of kernels");
            Gpu gpu = Gpu.Default;

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
            using (DeviceMemory2D<float>
                result_gpu = gpu.AllocateDevice<float>(h, finalWidth),
                source_gpu = gpu.AllocateDevice(source),
                kernels_gpu = gpu.AllocateDevice(kernels))
            {
                deviceptr<float>
                    presult_gpu = result_gpu.Ptr,
                    psource_gpu = source_gpu.Ptr,
                    pkernels_gpu = kernels_gpu.Ptr;
                int
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32(),
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    kernels_gpu_pitch = kernels_gpu.PitchInElements.ToInt32();

                // Backwards convolution
                void BackwardsKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / kDepth,         // Sample index
                        iKernelDepth = index % kDepth;    // Kernel index

                    // Process the convolution slice
                    int
                        targetBaseOffset = iSample * result_gpu_pitch + iKernelDepth * convolutionOutputSize,
                        sourceBaseOffset = iSample * source_gpu_pitch,
                        kernelBaseOffset = iKernelDepth * kSize;
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
                                    kernelDepthOffset = kernelBaseOffset + z * kernels_gpu_pitch;
                                for (int x = lowX; x <= highX; ++x)
                                {
                                    int
                                        sourceRowOffset = sourceDepthOffset + x * imgWidth,
                                        kernelRowOffset = kernelDepthOffset + (i - x) * kWidth + j;
                                    for (int y = lowY; y <= highY; ++y)
                                    {
                                        temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                    }
                                }
                            }
                            presult_gpu[targetRowOffset + j] = temp;
                        }
                    }
                }
                gpu.For(0, h * kDepth, BackwardsKernel);
                result_gpu.CopyToHost(out result);
            }
        }

        /// <summary>
        /// Performs a the gradient convolution operation on the source matrix, using the given kernels
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="sourceInfo">The source volume info (depth and 2D slices size)</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <param name="kernelsInfo">The kernels volume info (depth and 2D slices size)</param>
        /// <param name="result">The resulting matrix where each row contains the result of the convolutions for each original image for each sample</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        public static void ConvoluteGradient(
            in FloatSpan2D source, in VolumeInformation sourceInfo,
            in FloatSpan2D kernels, in VolumeInformation kernelsInfo,
            out FloatSpan2D result)
        {
            // Checks and local parameters
            int
                nKernels = kernels.Height,
                kw = kernels.Width,
                kDepth = kernelsInfo.Depth,
                kSize = kw / kernelsInfo.Depth,
                kHeight = kernelsInfo.Height,
                kWidth = kernelsInfo.Width;
            if (kHeight < 2 || kWidth < 2) throw new ArgumentException(nameof(kernels), "The kernel must be at least 2x2");
            int
                h = source.Height,
                w = source.Width,
                imgSize = sourceInfo.SliceSize,
                imgHeight = sourceInfo.Height,
                imgWidth = sourceInfo.Width;
            if (imgSize * sourceInfo.Depth != w) throw new ArgumentException(nameof(source), "Invalid depth parameter for the input matrix");
            if (imgSize < kSize) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            if (nKernels != h) throw new ArgumentException(nameof(kernels), "There must be a delta volume for each activation sample");
            Gpu gpu = Gpu.Default;

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
            using (DeviceMemory2D<float>
                result_gpu = gpu.AllocateDevice<float>(h, finalWidth),
                source_gpu = gpu.AllocateDevice(source),
                kernels_gpu = gpu.AllocateDevice(kernels))
            {
                deviceptr<float>
                    presult_gpu = result_gpu.Ptr,
                    psource_gpu = source_gpu.Ptr,
                    pkernels_gpu = kernels_gpu.Ptr;
                int
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32(),
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    kernels_gpu_pitch = kernels_gpu.PitchInElements.ToInt32();

                // Gradient kernel
                void GradientKernel(int index)
                {
                    // Calculate the current indexes
                    int
                        iSample = index / iterationsPerSample,      // Sample index
                        iMod = index % iterationsPerSample,
                        iSampleDepth = iMod / kDepth,               // Depth of the current gradient
                        iKernelDepth = iMod % kDepth;               // Output gradient index

                    // Process the current convolution slice
                    int
                        sourceBaseOffset = iSample * source_gpu_pitch + iSampleDepth * imgSize,
                        kernelBaseOffset = iSample * kernels_gpu_pitch + iKernelDepth * kSize,
                        resultBaseOffset = iSample * result_gpu_pitch + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;
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
                                    temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                }
                            }
                            presult_gpu[targetRowOffset + j] = temp;
                        }
                    }
                }
                gpu.For(0, h * iterationsPerSample, GradientKernel);
                result_gpu.CopyToHost(out result);
            }
        }

        /// <summary>
        /// Pools the input matrix with a window of 2 and a stride of 2
        /// </summary>
        /// <param name="source">The input matrix to pool</param>
        /// <param name="subdivision">The number of images in the data volume associated to each sample</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[,] Pool2x2([NotNull] this float[,] source, int subdivision)
        {
            // Checks
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (subdivision < 1) throw new ArgumentOutOfRangeException(nameof(subdivision), "The number of images per row can't be lower than 1");

            // Local parameters
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                imgSize = w % subdivision == 0 ? w / subdivision : throw new ArgumentException(nameof(source), "Invalid subdivision parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();          // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The width of the input matrix isn't valid");
            bool odd = imgAxis % 2 == 1;
            int
                inner = imgAxis - 1,    // Limit index for the edge cases
                scaledImageAxis = imgAxis / 2 + (odd ? 1 : 0),
                iterationsPerSample = subdivision * scaledImageAxis,
                scaledInner = scaledImageAxis - 1,
                scaledImageSize = scaledImageAxis * scaledImageAxis,
                finalWidth = scaledImageSize * subdivision;

            // Prepare the GPU memory
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float>
                source_gpu = gpu.AllocateDevice(source),
                result_gpu = gpu.AllocateDevice<float>(h, finalWidth))
            {
                // Pointers and pitches
                deviceptr<float>
                    psource_gpu = source_gpu.Ptr,
                    presult_gpu = result_gpu.Ptr;
                int
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32();

                // Pooling kernel
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / iterationsPerSample,        // Sample index
                        i_mod = index % iterationsPerSample,
                        j = i_mod / scaledImageAxis,            // Subdivision index
                        x_plain = i_mod % scaledImageAxis,
                        x = x_plain * 2;                        // Subdivision row index

                    // Assuming [x, y] are the indexes of the jth image for sample i
                    int
                        sample_offset = i * source_gpu_pitch,
                        image_offset = sample_offset + j * imgSize,
                        target_up = image_offset + x * imgAxis,
                        target_down = image_offset + (x + 1) * imgAxis,
                        result_offset = i * result_gpu_pitch + j * scaledImageSize + x_plain * scaledImageAxis;
                    if (x == inner)
                    {
                        // Last row
                        for (int y = 0; y < inner; y += 2)
                        {
                            int offset = target_up + y;
                            float
                                left = psource_gpu[offset],
                                right = psource_gpu[offset + 1],
                                max = left >= right ? left : right;
                            presult_gpu[result_offset + y / 2] = max;
                        }

                        // At this point the axis length must be an odd number
                        presult_gpu[result_offset + scaledInner] = psource_gpu[target_up + inner];
                    }
                    else
                    {
                        // Compute the maximum value of the current block
                        for (int y = 0; y < inner; y += 2)
                        {
                            int
                                up_offset = target_up + y,
                                down_offset = target_down + y;
                            float
                                upLeft = psource_gpu[up_offset],
                                upRight = psource_gpu[up_offset + 1],
                                downLeft = psource_gpu[down_offset],
                                downRight = psource_gpu[down_offset + 1],
                                upMax = upLeft >= upRight ? upLeft : upRight,
                                downMax = downLeft >= downRight ? downLeft : downRight,
                                max = upMax >= downMax ? upMax : downMax;
                            presult_gpu[result_offset + y / 2] = max;
                        }
                        if (odd)
                        {
                            float
                                up = psource_gpu[target_up + inner],
                                down = psource_gpu[target_down + inner],
                                max = up > down ? up : down;
                            presult_gpu[result_offset + scaledInner] = max;
                        }
                    }
                }

                // Process the pooling operation
                gpu.For(0, h * iterationsPerSample, Kernel);

                // Return the results
                Gpu.FreeAllImplicitMemory(true);
                return Gpu.Copy2DToHost(result_gpu);
            }
        }
    }
}
