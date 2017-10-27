using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Architecture;

namespace NeuralNetworkNET.Cuda.Helpers
{
    /// <summary>
    /// A static extension class that operates on matrices through Gpu computing
    /// </summary>
    public static class MatrixGpuExtensions
    {
        #region Multiplication

        /// <summary>
        /// Transposes the input matrix
        /// </summary>
        /// <param name="m">The matrix to transpose</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Transpose([NotNull] this double[,] m)
        {
            // Setup
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m_gpu = gpu.AllocateDevice(m))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(w, h))
            {
                // Local parameters
                deviceptr<double>
                    pm_gpu = m_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m_gpu_pitch = m_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();

                // Wrapper
                void Kernel(int ki)
                {
                    int offset = ki * m_gpu_pitch;
                    for (int j = 0; j < w; j++)
                        pmresult_gpu[j * mresult_gpu_pitch + ki] = pm_gpu[offset + j];
                }

                // Execute the multiplication in parallel
                gpu.For(0, h, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Multiply([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(h, w))
            {
                // Local parameters
                deviceptr<double>
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
                    double sum = 0;
                    int m1_offset = i * m1_gpu_pitch; // Constant within the loop
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1_gpu[m1_offset + k] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    pmresult_gpu[i * mresult_gpu_pitch + j] = sum;
                }

                // Execute the multiplication in parallel
                gpu.For(0, h * w, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices after transposing the first one
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] TransposeAndMultiply([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(0) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(l, w)) // The first matrix will be transposed
            {
                // Local parameters
                deviceptr<double>
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
                    double sum = 0;
                    for (int k = 0; k < h; k++)
                    {
                        sum += pm1_gpu[k * m1_gpu_pitch + i] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    pmresult_gpu[i * mresult_gpu_pitch + j] = sum;
                }

                // Execute the multiplication in parallel
                gpu.For(0, l * w, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        #endregion

        #region Combined

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the activation function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndActivation([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(h, w))
            {
                // Local parameters
                deviceptr<double>
                    pm1_gpu = m1_gpu.Ptr,
                    pm2_gpu = m2_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                    m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();
                Func<double, double> activation = ActivationFunctionProvider.Activation;

                // Wrapper
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / w,
                        j = index % w;

                    // Perform the multiplication
                    double sum = 0;
                    int m1_offset = i * m1_gpu_pitch; // Constant within the loop
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1_gpu[m1_offset + k] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    sum = activation(sum);
                    pmresult_gpu[i * mresult_gpu_pitch + j] = sum;
                }

                // Execute the multiplication in parallel
                gpu.For(0, h * w, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices and sums another vector to the result
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyWithSum([NotNull] this double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (m2.GetLength(1) != v.Length) throw new ArgumentException(nameof(v), "Invalid vector length");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory<double> v_gpu = gpu.AllocateDevice(v))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(h, w))
            {
                // Pointers and pitches
                deviceptr<double>
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
                    double sum = 0;
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
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the activation function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="v">The array to add to the resulting matrix before applying the activation function</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyWithSumAndActivation([NotNull] this double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
            if (m2.GetLength(1) != v.Length) throw new ArgumentException(nameof(v), "Invalid vector length");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            using (DeviceMemory<double> v_gpu = gpu.AllocateDevice(v))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(h, w))
            {
                // Pointers and pitches
                deviceptr<double>
                    pm1_gpu = m1_gpu.Ptr,
                    pm2_gpu = m2_gpu.Ptr,
                    pv_gpu = v_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                    m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();
                Func<double, double> activation = ActivationFunctionProvider.Activation;

                // Wrapper
                void Kernel(int index)
                {
                    // Calculate the current indexes
                    int
                        i = index / w,
                        j = index % w;

                    // Perform the multiplication
                    double sum = 0;
                    int pm1_offset = i * m1_gpu_pitch;
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1_gpu[pm1_offset + k] * pm2_gpu[k * m2_gpu_pitch + j];
                    }
                    sum += pv_gpu[j]; // Sum the input vector to each component;
                    pmresult_gpu[i * mresult_gpu_pitch + j] = activation(sum);
                }

                // Execute the multiplication in parallel
                gpu.For(0, h * w, Kernel);

                // Free memory and copy the result back
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Calculates d(L) by applying the Hadamard product of (yHat - y) and the activation prime of z
        /// </summary>
        /// <param name="a">The estimated y</param>
        /// <param name="y">The expected y</param>
        /// <param name="z">The activity on the last layer</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void InPlaceSubtractAndHadamardProductWithActivationPrime(
            [NotNull] this double[,] a, [NotNull] double[,] y, [NotNull] double[,] z)
        {
            // Checks
            int
                h = a.GetLength(0),
                w = a.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1) ||
                h != z.GetLength(0) || w != z.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Initialize the parameters and the result matrix
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> a_gpu = gpu.AllocateDevice(a))
            using (DeviceMemory2D<double> y_gpu = gpu.AllocateDevice(y))
            using (DeviceMemory2D<double> z_gpu = gpu.AllocateDevice(z))
            {
                // Pointers and pitches
                deviceptr<double>
                    pa_gpu = a_gpu.Ptr,
                    py_gpu = y_gpu.Ptr,
                    pz_gpu = z_gpu.Ptr;
                int
                    a_gpu_pitch = a_gpu.PitchInElements.ToInt32(),
                    y_gpu_pitch = y_gpu.PitchInElements.ToInt32(),
                    z_gpu_pitch = z_gpu.PitchInElements.ToInt32();
                Func<double, double> activation = ActivationFunctionProvider.ActivationPrime;

                // Wrapper
                void Kernel(int i)
                {
                    // Save the index and iterate for each column
                    int
                        y_gpu_offset = i * y_gpu_pitch,
                        z_gpu_offset = i * z_gpu_pitch;
                    for (int j = 0; j < w; j++)
                    {
                        int a_gpu_target = i * a_gpu_pitch + j;
                        double
                            difference = pa_gpu[a_gpu_target] - py_gpu[y_gpu_offset + j],
                            zPrime = activation(pz_gpu[z_gpu_offset + j]),
                            hProduct = difference * zPrime;
                        pa_gpu[a_gpu_target] = hProduct;
                    }
                }

                // Execute the multiplication in parallel
                gpu.For(0, h, Kernel);

                // Copy the results back
                Gpu.Copy2D(a_gpu, a);
            }
        }

        /// <summary>
        /// Calculates d(l) by applying the Hadamard product of d(l + 1) and W(l)T and the activation prime of z
        /// </summary>
        /// <param name="z">The activity on the previous layer</param>
        /// <param name="delta">The precomputed delta to use in the Hadamard product</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void InPlaceActivationPrimeAndHadamardProduct(
            [NotNull] this double[,] z, [NotNull] double[,] delta)
        {
            // Checks
            int
                h = z.GetLength(0),
                w = z.GetLength(1);
            if (h != delta.GetLength(0) || w != delta.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Initialize the parameters and the result matrix
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> z_gpu = gpu.AllocateDevice(z))
            using (DeviceMemory2D<double> d_gpu = Gpu.Default.AllocateDevice(delta))
            {
                // Pointers and pitches
                deviceptr<double>
                    pz_gpu = z_gpu.Ptr,
                    pd_gpu = d_gpu.Ptr;
                int
                    z_gpu_pitch = z_gpu.PitchInElements.ToInt32(),
                    d_gpu_pitch = d_gpu.PitchInElements.ToInt32();
                Func<double, double> activation = ActivationFunctionProvider.ActivationPrime;

                // Wrapper
                void Kernel(int i)
                {
                    // Save the index and iterate for each column
                    int
                        pz_offset = i * z_gpu_pitch,
                        pd_offset = i * d_gpu_pitch;
                    for (int j = 0; j < w; j++)
                    {
                        int pz_target = pz_offset + j;
                        pz_gpu[pz_target] = activation(pz_gpu[pz_target]) * pd_gpu[pd_offset + j];
                    }
                }

                // Execute the multiplication in parallel
                gpu.For(0, h, Kernel);

                // Copy the results back
                Gpu.Copy2D(z_gpu, z);
            }
        }

        #endregion

        #region Misc

        /// <summary>
        /// Performs the activation function on the input matrix
        /// </summary>
        /// <param name="m">The input matrix</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] Activation([NotNull] this double[,] m)
        {
            // Setup
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m_gpu = gpu.AllocateDevice(m))
            using (DeviceMemory2D<double> mresult_gpu = gpu.AllocateDevice<double>(h, w))
            {
                // Local parameters
                deviceptr<double>
                    pm_gpu = m_gpu.Ptr,
                    pmresult_gpu = mresult_gpu.Ptr;
                int
                    m_gpu_pitch = m_gpu.PitchInElements.ToInt32(),
                    mresult_gpu_pitch = mresult_gpu.PitchInElements.ToInt32();
                Func<double, double> activation = ActivationFunctionProvider.Activation;

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
                return Gpu.Copy2DToHost(mresult_gpu);
            }
        }

        /// <summary>
        /// Calculates half the sum of the squared difference of each value pair in the two matrices
        /// </summary>
        /// <param name="m1">The first matrix</param>
        /// <param name="m2">The second matrix</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double HalfSquaredDifference([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Detect the size of the inputs
            int h = m1.GetLength(0), w = m1.GetLength(1);
            if (h != m2.GetLength(0) || w != m2.GetLength(1)) throw new ArgumentException("The two matrices must have the same size");

            // Allocate parameters
            Gpu gpu = Gpu.Default;
            using (DeviceMemory2D<double> m1_gpu = gpu.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_gpu = gpu.AllocateDevice(m2))
            {
                // Check Compute Capability (64bit atomicAdd function requires Compute 6.x)
                if (gpu.Device.Arch.Major >= 6)
                {
                    using (DeviceMemory<double> result_gpu = gpu.AllocateDevice<double>(1))
                    {
                        // Local parameters
                        deviceptr<double>
                            pm1_gpu = m1_gpu.Ptr,
                            pm2_gpu = m2_gpu.Ptr,
                            presult_gpu = result_gpu.Ptr;
                        int
                            m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                            m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32();

                        // Wrapper
                        void Kernel(int i)
                        {
                            // Local sum over each row
                            double row = 0;
                            int
                                m1_offset = i * m1_gpu_pitch,
                                m2_offset = i * m2_gpu_pitch;
                            for (int j = 0; j < w; j++)
                            {
                                double delta = pm1_gpu[m1_offset + j] - pm2_gpu[m2_offset + j];
                                row += delta * delta;
                            }
                            DeviceFunction.AtomicAdd(presult_gpu, row);
                        }

                        // Compute the total sum
                        gpu.For(0, h, Kernel);

                        // Return the result
                        double[] result = Gpu.CopyToHost(result_gpu);
                        return result[0] / 2;
                    }
                }

                // Legacy code
                using (DeviceMemory<double> result_gpu = gpu.AllocateDevice<double>(h))
                {
                    // Local parameters
                    deviceptr<double>
                        pm1_gpu = m1_gpu.Ptr,
                        pm2_gpu = m2_gpu.Ptr,
                        presult_gpu = result_gpu.Ptr;
                    int
                        m1_gpu_pitch = m1_gpu.PitchInElements.ToInt32(),
                        m2_gpu_pitch = m2_gpu.PitchInElements.ToInt32();

                    // Wrapper
                    void Kernel(int i)
                    {
                        // Local sum over each row
                        double row = 0;
                        int
                            m1_offset = i * m1_gpu_pitch,
                            m2_offset = i * m2_gpu_pitch;
                        for (int j = 0; j < w; j++)
                        {
                            double delta = pm1_gpu[m1_offset + j] - pm2_gpu[m2_offset + j];
                            row += delta * delta;
                        }
                        presult_gpu[i] = row;
                    }

                    // Compute the total sum
                    gpu.For(0, h, Kernel);
                    double AggregateKernel(double a, double b) => a + b;
                    return gpu.Aggregate(presult_gpu, h, AggregateKernel) / 2;
                }
            }
        }

        /// <summary>
        /// Compresses a matrix into a row vector by summing the components column by column
        /// </summary>
        /// <param name="m">The matrix to compress</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[] CompressVertically([NotNull] this double[,] m)
        {
            // Setup
            Gpu gpu = Gpu.Default;
            int
                h = m.GetLength(0),
                w = m.GetLength(1);
            using (DeviceMemory2D<double> m_gpu = gpu.AllocateDevice(m))
            using (DeviceMemory<double> vresult_gpu = gpu.AllocateDevice<double>(w))
            {
                // Pointers
                deviceptr<double>
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
                        double sum = 0;
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

        #endregion
    }
}
