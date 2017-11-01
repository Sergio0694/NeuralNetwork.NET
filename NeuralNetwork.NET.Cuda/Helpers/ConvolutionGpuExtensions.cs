using System;
using System.Linq;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// A static class that contains some GPU-accelerated convolution extension methods
    /// </summary>
    public static class ConvolutionGpuExtensions
    {
        /// <summary>
        /// Performs a 3*3 convolution on the source matrix, using the given kernel, in parallel
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="subdivision">The number of images in the data volume associated to each sample</param>
        /// <param name="kernels">The list of convolution kernels to apply to each image</param>
        /// <returns>A new matrix where each row contains the result of the convolutions for each original image for each sample</returns>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid, or the kernels list isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Convolute3x3([NotNull] this double[,] source, int subdivision, [NotNull]  params double[][,] kernels)
        {
            // Checks
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (subdivision < 1) throw new ArgumentOutOfRangeException(nameof(subdivision), "The number of images per row can't be lower than 1");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels list can't be empty");
            if (kernels.Any(k => k.GetLength(0) != 3 || k.GetLength(1) != 3))
                throw new ArgumentException(nameof(kernels), "One of the input kernels doesn't have a valid size");

            // Local parameters
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                klen = kernels.Length,
                imgSize = w % subdivision == 0 ? w / subdivision : throw new ArgumentException(nameof(source), "Invalid subdivision parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The size of the input matrix isn't valid");
            if (imgSize < 9) throw new ArgumentOutOfRangeException("Each subdivided matrix must at least have the size of the kernels");
            int
                inner = imgAxis - 2,                                            // Size of each image edge after the convolution
                convolutionOutputSize = inner * inner,                          // Size of each processed image
                finalWidth = inner * inner * subdivision * kernels.Length,      // Final size of each sample row
                iterationsPerSubdivision = klen * inner,                        // GPU iterations for each sample sub-image
                iterationsPerSample = subdivision * iterationsPerSubdivision;   // GPU iterations for each dataset entry

            // Prepare the kernels info
            double[] norms = new double[klen];

            // Precompute the normalization factors
            unsafe
            {
                fixed (double* pnorms = norms)
                {
                    for (int i = 0; i < klen; i++)
                    {
                        fixed (double* pk = kernels[i])
                        {
                            double factor = 0;
                            for (int j = 0; j < 3; j++)
                                for (int k = 0; k < 3; k++)
                                    factor += pk[j * 3 + k].Abs();
                            pnorms[i] = factor;
                        }
                    }
                }
            }

            // Process the convolution in parallel
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> source_gpu = gpu.AllocateDevice(source))
            using (DeviceMemory2D<double> kernels_gpu = gpu.AllocateDevice<double>(kernels.Length, 9))  // Series of 3*3 kernels
            using (DeviceMemory<double> norms_gpu = gpu.AllocateDevice(norms))
            using (DeviceMemory2D<double> result_gpu = gpu.AllocateDevice<double>(h, finalWidth))
            {
                // Pointers and pitches
                deviceptr<double>
                    psource_gpu = source_gpu.Ptr,
                    pkernels_gpu = kernels_gpu.Ptr,
                    pnorms_gpu = norms_gpu.Ptr,
                    presult_gpu = result_gpu.Ptr;
                int
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    kernels_gpu_pitch = kernels_gpu.PitchInElements.ToInt32(),
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32();

                // Copy the kernels to GPU memory
                for (int i = 0; i < klen; i++)
                {
                    Gpu.Copy(kernels[i], 0, gpu, pkernels_gpu + i * kernels_gpu_pitch, 9);
                }

                // Convolution kernel
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / iterationsPerSample,            // Sample index
                        i_mod = index % iterationsPerSample,
                        j = i_mod / iterationsPerSubdivision,       // Subdivision index
                        j_mod = i_mod % iterationsPerSubdivision,
                        k = j_mod / inner,                          // Kernel index
                        x = j_mod % inner;                          // Sub-image x index

                    // Assuming [x, y] are the indexes of the jth image for sample i, applying kernel k
                    int
                        sample_offset = i * source_gpu_pitch,
                        image_offset = sample_offset + j * imgSize,
                        kernel_offset = k * kernels_gpu_pitch,
                        base_upper_offset = image_offset + x * imgAxis,
                        base_middle_offset = image_offset + (x + 1) * imgAxis,
                        base_lower_offset = image_offset + (x + 2) * imgAxis,
                        base_target_offset =
                            i * result_gpu_pitch +
                            j * convolutionOutputSize * klen +
                            k * convolutionOutputSize +
                            x * inner;

                    // Iterate over the columns in the current row and process the convolution
                    for (int y = 0; y < inner; y++)
                    {
                        int
                            upper_offset = base_upper_offset + y,
                            middle_offset = base_middle_offset + y,
                            lower_offset = base_lower_offset + y;
                        double
                            partial =
                                psource_gpu[upper_offset] * pkernels_gpu[kernel_offset] +
                                psource_gpu[upper_offset + 1] * pkernels_gpu[kernel_offset + 1] +
                                psource_gpu[upper_offset + 2] * pkernels_gpu[kernel_offset + 2] +
                                psource_gpu[middle_offset] * pkernels_gpu[kernel_offset + 3] +
                                psource_gpu[middle_offset + 1] * pkernels_gpu[kernel_offset + 4] +
                                psource_gpu[middle_offset + 2] * pkernels_gpu[kernel_offset + 5] +
                                psource_gpu[lower_offset] * pkernels_gpu[kernel_offset + 6] +
                                psource_gpu[lower_offset + 1] * pkernels_gpu[kernel_offset + 7] +
                                psource_gpu[lower_offset + 2] * pkernels_gpu[kernel_offset + 8],
                            normalized = partial / pnorms_gpu[k];
                        presult_gpu[base_target_offset + y] = normalized;
                    }
                }

                // Convolute in parallel
                gpu.For(0, h * iterationsPerSample, Kernel);

                // Return the processed results
                return Gpu.Copy2DToHost(result_gpu);
            }
        }

        /// <summary>
        /// Performs the in place Rectified Linear Units operation on the input matrix (applies a minimum value of 0)
        /// </summary>
        /// <param name="m">The input matrix to edit</param>
        /// <remarks>This method is still executed on the CPU, as it only costs O(n^2) and more parallelization wouldn't help</remarks>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void ReLU([NotNull] this double[,] m)
        {
            int h = m.GetLength(0), w = m.GetLength(1);
            bool result = Parallel.For(0, h, i =>
            {
                unsafe
                {
                    fixed (double* p = m)
                    {
                        int offset = i * w;
                        for (int j = 0; j < w; j++)
                        {
                            int target = offset + j;
                            if (p[target] < 0) p[target] = 0;
                        }
                    }
                }
            }).IsCompleted;
            if (!result) throw new InvalidOperationException("There was an error while executing the operation");
        }

        /// <summary>
        /// Performs a normalization on the input matrix, considering the results of a previous convolution, in parallel
        /// </summary>
        /// <param name="source">The source matrix, where each row is a sample in the dataset and each one contains a series of images in row-first order</param>
        /// <param name="subdivision">The number of images in the data volume associated to each sample</param>
        /// <exception cref="ArgumentException">The size of the matrix isn't valid</exception>
        /// <exception cref="ArgumentOutOfRangeException">The size of the matrix doesn't match the expected values</exception>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.ModifyExistingContent)]
        public static void Normalize([NotNull] this double[,] source, int subdivision)
        {
            // Checks
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (subdivision < 1) throw new ArgumentOutOfRangeException(nameof(subdivision), "The number of images per row can't be lower than 1");

            // Local parameters
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                iterations = h * subdivision,
                imgSize = w % subdivision == 0 ? w / subdivision : throw new ArgumentException(nameof(source), "Invalid subdivision parameter for the input matrix"),
                imgAxis = imgSize.IntegerSquare();  // Size of an edge of one of the inner images per sample
            if (imgAxis * imgAxis != imgSize) throw new ArgumentOutOfRangeException(nameof(source), "The width of the input matrix isn't valid");

            // Process the convolution in parallel
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> source_gpu = gpu.AllocateDevice(source))
            using (DeviceMemory<double> norms_gpu = gpu.AllocateDevice<double>(h * subdivision))    // Vector to store the normalization factors
            {
                // Pointers and pitches
                deviceptr<double>
                    psource_gpu = source_gpu.Ptr,
                    pnorms_gpu = norms_gpu.Ptr;
                int source_gpu_pitch = source_gpu.PitchInElements.ToInt32();

                // Precalculation kernel
                void Kernel0(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / subdivision,    // Sample index
                        j = index % subdivision,    // Subdivision index
                        sample_offset = i * source_gpu_pitch,
                        image_offset = sample_offset + j * imgSize;

                    // Compute the current normalization factor
                    double sum = 0;
                    for (int k = 0; k < imgSize; k++)
                        sum += psource_gpu[image_offset + k];
                    pnorms_gpu[i * j] = sum;
                }

                // Normalization kernel
                void Kernel1(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / subdivision,    // Sample index
                        j = index % subdivision,    // Subdivision index
                        sample_offset = i * source_gpu_pitch,
                        image_offset = sample_offset + j * imgSize;

                    // Normalize the sub-matrix
                    double factor = pnorms_gpu[i * j];
                    for (int k = 0; k < imgSize; k++)
                        psource_gpu[image_offset + k] /= factor;
                }

                // Start the normalization
                gpu.For(0, iterations, Kernel0);
                gpu.For(0, iterations, Kernel1);

                // Copy the results back
                Gpu.Copy2D(source_gpu, source);
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
        public static double[,] Pool2x2([NotNull] this double[,] source, int subdivision)
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
            using (DeviceMemory2D<double> source_gpu = gpu.AllocateDevice(source))
            using (DeviceMemory2D<double> result_gpu = gpu.AllocateDevice<double>(h, finalWidth))
            {
                // Pointers and pitches
                deviceptr<double>
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
                            double
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
                            double
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
                            double
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
                return Gpu.Copy2DToHost(result_gpu);
            }
        }
    }
}
