using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Convolution
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
        public static float[][,] Convolute3x3([NotNull] this float[,] source, int subdivision, [NotNull]  params float[][,] kernels)
        {
            throw new NotImplementedException();
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
