using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Convolution
{
    /// <summary>
    /// A static class that contains some GPU-accelerated convolution extension methods
    /// </summary>
    public static class ConvolutionGpuExtensions
    {
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
            Gpu gpu = Gpu.Default;

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
                                            temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                        }
                                    }
                                }
                                presult_gpu[targetRowOffset + j] = temp;
                            }
                        }
                    }
                    gpu.For(0, h * nKernels, ForwardKernel);
                    return Gpu.Copy2DToHost(result_gpu);
                }
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
                            iSample = index / nKernels,     // Sample index
                            iKernel = index % nKernels;     // Kernel index

                        // Process the convolution slice
                        int
                            targetBaseOffset = iSample * result_gpu_pitch + iKernel * convolutionOutputSize,
                            sourceBaseOffset = iSample * source_gpu_pitch,
                            kernelBaseOffset = iKernel * kernels_gpu_pitch;
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
                                            temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                        }
                                    }
                                }
                                presult_gpu[targetRowOffset + j] = temp;
                            }
                        }
                    }
                    gpu.For(0, h * nKernels, BackwardsKernel);
                    return Gpu.Copy2DToHost(result_gpu);
                }
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
                            iSampleDepth = iMod / kernelsDepth,         // Depth of the current gradient
                            iKernelDepth = iMod % kernelsDepth;         // Output gradient index

                        // Process the current convolution slice
                        int
                            sourceBaseOffset = iSample * source_gpu_pitch + iSampleDepth * imgSize,
                            kernelBaseOffset = iSample * kernels_gpu_pitch + iKernelDepth * kSize,
                            resultBaseOffset = iSample * result_gpu_pitch + iKernelDepth * gradientSize + iSampleDepth * convolutionOutputSize;
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
                                        temp += psource_gpu[sourceRowOffset + y] * pkernels_gpu[kernelRowOffset - y];
                                    }
                                }
                                presult_gpu[targetRowOffset + j] = temp;
                            }
                        }
                    }
                    gpu.For(0, h * iterationsPerSample, GradientKernel);
                    return Gpu.Copy2DToHost(result_gpu);
                }
            }
            throw new ArgumentOutOfRangeException(nameof(mode), "Unsupported convolution mode");
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
