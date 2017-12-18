using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Extensions
{
    /// <summary>
    /// A static extension class that operates on matrices through Gpu computing
    /// </summary>
    internal static class Blas
    {
        /// <summary>
        /// Performs the multiplication between two matrices after transposing the first one
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="result">The resulting matrix</param>
        public static void TransposeAndMultiply(in Tensor m1, in Tensor m2, out Tensor result)
        {
            // Checks
            if (m1.Entities != m2.Entities) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.Entities;
            int w = m2.Length;
            int l = m1.Length;
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float> 
                m1_gpu = gpu.AllocateDevice2D(m1),
                m2_gpu = gpu.AllocateDevice2D(m2),
                mresult_gpu = gpu.AllocateDevice<float>(l, w)) // The first matrix will be transposed
            {
                // Local parameters
                deviceptr<float>
                    pm1_gpu = m1_gpu.Ptr,
                    pm2_gpu = m2_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                    m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();

                // Wrapper
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / w,
                        j = index % w;

                    // Perform the multiplication
                    float sum = 0;
                    for (int k = 0; k < h; k++)
                    {
                        sum += pm1_gpu[k * m1_gpu_pitch + i] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    pmresult_gpu[i * mresult_gpu_pitch + j] = sum;
                }

                // Execute the multiplication in parallel
                gpu.For(0, l * w, Kernel);

                // Return the result
                mresult_gpu.CopyToHost(out result);
            }
        }

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
