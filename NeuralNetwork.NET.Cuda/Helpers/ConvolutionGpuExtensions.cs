using System;
using System.Linq;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// A static class that contains some GPU-accelerated convolution extension methods
    /// </summary>
    internal static class ConvolutionGpuExtensions
    {
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Convolute3x3([NotNull] this double[,] source, [NotNull]  params double[][,] kernels)
        {
            // Checks
            if (source.Length == 0) throw new ArgumentException(nameof(source), "The source matrix can't be empty");
            if (kernels.Length == 0) throw new ArgumentException(nameof(kernels), "The kernels list can't be empty");
            if (kernels.Any(k => k.GetLength(0) != 3 || k.GetLength(1) != 3))
                throw new ArgumentException(nameof(kernels), "One of the input kernels doesn't have a valid size");

            // Local parameters
            int
                h = source.GetLength(0),
                w = source.GetLength(1),
                klen = kernels.Length,
                size = w.IntegerSquare();
            if (size * size != w) throw new ArgumentOutOfRangeException(nameof(source), "The width of the input matrix isn't valid");
            int
                inner = size - 1,
                wf = inner * inner * kernels.Length;

            // Prepare the kernels info
            double[] norms = new double[klen];

            // Calculate the normalization factor
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
            using (DeviceMemory<double> norms_gpu = gpu.AllocateDevice(norms))
            using (DeviceMemory2D<double> result_gpu = gpu.AllocateDevice<double>(h, wf))
            {
                // Pointers and pitches
                deviceptr<double>
                    psource_gpu = source_gpu.Ptr,
                    presult_gpu = result_gpu.Ptr;
                int
                    source_gpu_pitch = source_gpu.PitchInElements.ToInt32(),
                    result_gpu_pitch = result_gpu.PitchInElements.ToInt32();

                // Convolution kernel
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / klen,   // Row (image) index
                        j = index % klen;   // Kernel number

                    int img_offset = i * source_gpu_pitch;
                    for (int x = 1; x < size - 1; x++)
                    {
                        for (int y = 1; y < size - 1; y++)
                        {
                            /* 
                            double
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
                            result[x, y++] = normalized; */
                        }
                    }
                }

                // Convolute in parallel
                gpu.For(0, h * klen, Kernel);

                // Return the processed results
                return Gpu.Copy2DToHost(result_gpu);
            }
        }
    }
}
