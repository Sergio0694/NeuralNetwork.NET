using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// A static extension class that operates on matrices through Gpu computing
    /// </summary>
    internal static class MatrixGpuExtensions
    {
        /// <summary>
        /// Performs the multiplication between two matrices after transposing the first one
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="result">The resulting matrix</param>
        public static void TransposeAndMultiply(in FloatSpan2D m1, in FloatSpan2D m2, out FloatSpan2D result)
        {
            // Checks
            if (m1.Height != m2.Height) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.Height;
            int w = m2.Width;
            int l = m1.Width;
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<float> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory2D<float> mresult_gpu = gpu.AllocateDevice<float>(l, w)) // The first matrix will be transposed
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
        /// Performs the multiplication between two matrices and sums another vector to the result
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix</param>
        /// <param name="result">The resulting matrix</param>
        public static void MultiplyWithSum(in FloatSpan2D m1, float[,] m2, float[] v, out FloatSpan2D result)
        {
            // Checks
            if (m1.Width != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (m2.GetLength(1) != v.Length) throw new ArgumentException(nameof(v), "Invalid vector length");

            // Initialize the parameters and the result matrix
            int h = m1.Height;
            int w = m2.GetLength(1);
            int l = m1.Width;
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<float> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory<float> v_gpu = gpu.AllocateDevice(v))
            using (DeviceMemory2D<float> mresult_gpu = gpu.AllocateDevice<float>(h, w))
            {
                // Pointers and pitches
                deviceptr<float>
                    pm1_gpu = m1_gpu.Ptr,
                    pm2_gpu = m2_gpu.Ptr,
                    pv_gpu = v_gpu.Ptr,
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
                    int pm1_offset = i * m1_gpu_pitch;
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1_gpu[pm1_offset + k] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    pmresult_gpu[i * mresult_gpu_pitch + j] = sum + pv_gpu[j]; // Sum the input vector to each component
                }

                // Execute the multiplication in parallel
                gpu.For(0, h * w, Kernel);

                // Free memory and copy the result back
                mresult_gpu.CopyToHost(out result);
            }
        }

        /// <summary>
        /// Calculates d(l) by applying the Hadamard product of d(l + 1) and W(l)T and the activation prime of z
        /// </summary>
        /// <param name="z">The activity on the previous layer</param>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="prime">The activation prime function to use</param>
        public static void MultiplyAndHadamardProductWithActivation(
            in FloatSpan2D z, in FloatSpan2D m1, in FloatSpan2D m2, [NotNull] ActivationFunction prime)
        {
            // Initialize the parameters and the result matrix
            int h = m1.Height;
            int w = m2.Width;
            int l = m1.Width;

            // Checks
            if (l != m2.Height) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (h != z.Height || w != z.Width) throw new ArgumentException("The matrices must be of equal size");

            // Initialize the parameters and the result matrix
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float> z_gpu = gpu.AllocateDevice(z))
            using (DeviceMemory2D<float> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<float> m2_gpu = gpu.AllocateDevice(m2))
            {
                // Pointers and pitches
                deviceptr<float>
                    pz_gpu = z_gpu.Ptr,
                    pm1_gpu = m1_gpu.Ptr,
                    pm2_gpu = m2_gpu.Ptr;
                int
                    z_gpu_pitch = z_gpu.PitchInElements.ToInt32(),
                    m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                    m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32();

                // Wrapper
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / w,
                        j = index % w;

                    // Perform the multiplication
                    float sum = 0;
                    int m1_offset = i * m1_gpu_pitch; // Constant within the loop
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1_gpu[m1_offset + k] * pm2_gpu[k * m2_gpu_pitch + j];
                    }

                    // sum is now the final delta(l) value in position [i, j]
                    int z_target = i * z_gpu_pitch + j;
                    pz_gpu[z_target] = prime(pz_gpu[z_target]) * sum;
                }

                // Execute the multiplication in parallel
                gpu.For(0, h * w, Kernel);

                // Copy the results back
                z_gpu.CopyTo(z);
            }
        }

        /// <summary>
        /// Performs the activation function on the input matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        /// <param name="activation">The activation function to use</param>
        /// <param name="result">The resulting matrix</param>
        public static void Activation(in FloatSpan2D m, [NotNull] ActivationFunction activation, out FloatSpan2D result)
        {
            // Setup
            int
                h = m.Height,
                w = m.Width;
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<float> m_gpu = gpu.AllocateDevice(m))
            using (DeviceMemory2D<float> mresult_gpu = gpu.AllocateDevice<float>(h, w))
            {
                // Local parameters
                deviceptr<float>
                    pm_gpu = m_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m_gpu_pitch = m_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();

                // Wrapper
                void Kernel(int ki)
                {
                    int
                        m1_offset = ki * m_gpu_pitch,
                        mresult_offset = ki * mresult_gpu_pitch;
                    for (int j = 0; j < w; j++)
                        pmresult_gpu[mresult_offset + j] = activation(pm_gpu[m1_offset + j]);
                }

                // Execute the multiplication in parallel
                gpu.For(0, h, Kernel);

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
