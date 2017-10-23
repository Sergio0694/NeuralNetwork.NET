using System;
using Alea;
using Alea.Parallel;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Cuda
{
    public static class GpuPreferences
    {
        private static ProcessingMode _ProcessingMode = ProcessingMode.Cpu;

        public static ProcessingMode ProcessingMode
        {
            get => _ProcessingMode;
            set
            {
                if (_ProcessingMode != value)
                {
                    switch (value)
                    {
                        case ProcessingMode.Cpu:
                            MatrixExtensions.MultiplyOverride = null;
                            break;
                        case ProcessingMode.Gpu:
                            MatrixExtensions.MultiplyOverride = CudaMultiply;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(value), value, null);
                    }
                    _ProcessingMode = value;
                }
            }
        }
        
        private static double[,] CudaMultiply(double[,] m1, double[,] m2)
        {
            int m = m1.GetLength(0);
            int n = m2.GetLength(1);
            int p = m2.GetLength(1);
            double[,] result = new double[m, p];
            

            Gpu.Default.For(0, m * p, index =>
            {
                int
                    i = index / p,
                    j = index % p;

                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += m1[i, k] * m2[k, j];
                }
                result[i, j] = sum;
            });
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the sigmoid function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [System.Diagnostics.Contracts.Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndSigmoid([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);

            // Execute the multiplication in parallel
            using (DeviceMemory2D<double> m1_device = Gpu.Default.AllocateDevice(m1))
            using (DeviceMemory2D<double> m2_device = Gpu.Default.AllocateDevice(m2))
            using (DeviceMemory2D<double> mresult_device = Gpu.Default.AllocateDevice<double>(h, w))
            {
                // Pointers setup
                deviceptr<double>
                    pm1 = m1_device.Ptr,
                    pm2 = m2_device.Ptr,
                    pmresult = mresult_device.Ptr;

                // Local wrapper function
                void Kernel(int ki)
                {
                    // Calculate the current indexes
                    int
                        i = ki / w,
                        j = ki % w;

                    // Perform the multiplication
                    double sum = 0;
                    for (int k = 0; k < l; k++)
                    {
                        sum += pm1[i * w + k] * pm2[k * l + j];
                    }
                    double sigmoid = 1 / (1 + Math.Exp(-sum));
                    pmresult[i * w + j] = sigmoid;
                }

                // Get the pointers and iterate fo each row
                Gpu.Default.For(0, h * w, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_device);
            }
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the sigmoid function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [System.Diagnostics.Contracts.Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndSigmoid2([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int m = m1.GetLength(0);
            int n = m2.GetLength(1);
            int p = m1.GetLength(1);
            double[,]
                m1_gpu = Gpu.Default.Allocate(m1),
                m2_gpu = Gpu.Default.Allocate(m2),
                mresult_gpu = Gpu.Default.Allocate<double>(m, p);

            // Execute the multiplication in parallel
            Gpu.Default.For(0, m * p, index =>
            {
                // Calculate the current indexes
                int
                    i = index / p,
                    j = index % p;

                // Perform the multiplication
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += m1_gpu[i, k] * m2_gpu[k, j];
                }
                double sigmoid = 1 / (1 + Math.Exp(-sum));
                mresult_gpu[i, j] = sigmoid;
            });

            // Free memory and copy the result back
            Gpu.Free(m1_gpu);
            Gpu.Free(m2_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
        }
    }
}
