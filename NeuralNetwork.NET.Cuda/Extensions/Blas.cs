using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Cuda.Extensions
{
    /// <summary>
    /// A static extension class that operates on matrices through Gpu computing
    /// </summary>
    internal static class Blas
    {
        /// <summary>
        /// Compresses a matrix into a row vector by summing the components column by column
        /// </summary>
        /// <param name="m">The matrix to compress</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static float[] CompressVertically([NotNull] this float[,] m)
        {
            // Setup
            Gpu gpu = Gpu.Default;
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            using (DeviceMemory2D<float> m_gpu = gpu.AllocateDevice(m))
            using (DeviceMemory<float> vresult_gpu = gpu.AllocateDevice<float>(w))
            {
                // Pointers
                deviceptr<float>
                    pm_gpu = m_gpu.Ptr,
                    pvresult_gpu = vresult_gpu.Ptr;
                int pitch = m_gpu.PitchInElements.ToInt32();

                // Check Compute Capability (64bit atomicAdd function requires Compute 6.x)
                if (gpu.Device.Arch.Major >= 6)
                {
                    // Wrapper 
                    void Kernel(int ki)
                    {
                        int m_gpu_offset = ki * pitch;
                        for (int j = 0; j < w; j++)
                            DeviceFunction.AtomicAdd(pvresult_gpu + j, pm_gpu[m_gpu_offset + j]);
                    }

                    // Execute the multiplication for GPUs with Compute Capability >= 6.x
                    gpu.For(0, h, Kernel);
                }
                else
                {
                    // Legacy wrapper 
                    void Kernel(int kj)
                    {
                        float sum = 0;
                        for (int i = 0; i < h; i++)
                            sum += pm_gpu[i * pitch + kj];
                        pvresult_gpu[kj] = sum;
                    }

                    // Execute the multiplication for GPUs with Compute Capability >= 6.x
                    gpu.For(0, w, Kernel);
                }

                // Return the results
                return Gpu.CopyToHost(vresult_gpu);
            }
        }
    }
}
