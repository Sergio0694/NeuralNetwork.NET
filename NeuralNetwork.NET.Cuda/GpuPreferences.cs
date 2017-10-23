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
                         //   MatrixExtensions.MultiplyOverride = CudaMultiply;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(value), value, null);
                    }
                    _ProcessingMode = value;
                }
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
            double[,]
                m1_gpu = Gpu.Default.Allocate(m1),
                m2_gpu = Gpu.Default.Allocate(m2),
                mresult_gpu = Gpu.Default.Allocate<double>(h, w);

            // Wrapper
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    i = index / w,
                    j = index % w;

                // Perform the multiplication
                double sum = 0;
                for (int k = 0; k < l; k++)
                {
                    sum += m1_gpu[i, k] * m2_gpu[k, j];
                }
                mresult_gpu[i, j] = sum;
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h * w, Kernel);

            // Free memory and copy the result back
            Gpu.Free(m1_gpu);
            Gpu.Free(m2_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the sigmoid function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndSigmoid([NotNull] this double[,] m1, [NotNull] double[,] m2)
        {
            // Checks
            if (m1.GetLength(1) != m2.GetLength(0)) throw new ArgumentOutOfRangeException("Invalid matrices sizes");

            // Initialize the parameters and the result matrix
            int h = m1.GetLength(0);
            int w = m2.GetLength(1);
            int l = m1.GetLength(1);
            double[,]
                m1_gpu = Gpu.Default.Allocate(m1),
                m2_gpu = Gpu.Default.Allocate(m2),
                mresult_gpu = Gpu.Default.Allocate<double>(h, w);

            // Wrapper
            void Kernel(int index)
            {
                // Calculate the current indexes
                int
                    i = index / w,
                    j = index % w;

                // Perform the multiplication
                double sum = 0;
                for (int k = 0; k < l; k++)
                {
                    sum += m1_gpu[i, k] * m2_gpu[k, j];
                }
                double sigmoid = 1 / (1 + Math.Exp(-sum));
                mresult_gpu[i, j] = sigmoid;
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h * w, Kernel);

            // Free memory and copy the result back
            Gpu.Free(m1_gpu);
            Gpu.Free(m2_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
        }

        /// <summary>
        /// Calculates d(L) by applying the Hadamard product of (yHat - y) and the sigmoid prime of z
        /// </summary>
        /// <param name="a">The estimated y</param>
        /// <param name="y">The expected y</param>
        /// <param name="z">The activity on the last layer</param>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static void InPlaceSubtractAndHadamardProductWithSigmoidPrime(
            [NotNull] this double[,] a, [NotNull] double[,] y, [NotNull] double[,] z)
        {
            // Checks
            int
                h = a.GetLength(0),
                w = a.GetLength(1);
            if (h != y.GetLength(0) || w != y.GetLength(1) ||
                h != z.GetLength(0) || w != z.GetLength(1)) throw new ArgumentException("The matrices must be of equal size");

            // Initialize the parameters and the result matrix
            double[,] 
                a_gpu = Gpu.Default.Allocate(a),
                y_gpu = Gpu.Default.Allocate(y),
                z_gpu = Gpu.Default.Allocate(z);

            // Wrapper
            void Kernel(int i)
            {
                // Save the index and iterate for each column
                for (int j = 0; j < w; j++)
                {
                    double
                        difference = a_gpu[i, j] - y_gpu[i, j],
                        exp = Math.Exp(-z_gpu[i, j]),
                        sum = 1 + exp,
                        square = sum * sum,
                        zPrime = exp / square,
                        hProduct = difference * zPrime;
                    a_gpu[i, j] = hProduct;
                }
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h, Kernel);

            // Free memory and copy the result back
            Gpu.Free(y_gpu);
            Gpu.Free(z_gpu);
            Gpu.Copy(a_gpu, a);
            Gpu.Free(a_gpu);
        }

        /// <summary>
        /// Performs the multiplication between two matrices and then applies the sigmoid function
        /// </summary>
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static double[,] MultiplyAndSigmoidOnDeviceMemory([NotNull] this double[,] m1, [NotNull] double[,] m2)
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
                    int im1 = i * l;
                    for (int k = 0; k < l; k++)
                    {
                        // m1[i, k] * m2[k, j]
                        sum += pm1[im1 + k] * pm2[k * w + j];
                    }
                    double sigmoid = 1 / (1 + Math.Exp(-sum)); // Apply the sigmoid
                    pmresult[i * w + j] = sigmoid; // result[i, j]
                }

                // Get the pointers and iterate fo each row
                Gpu.Default.For(0, h * w, Kernel);

                // Return the result
                return Gpu.Copy2DToHost(mresult_device);
            }
        }
    }
}
