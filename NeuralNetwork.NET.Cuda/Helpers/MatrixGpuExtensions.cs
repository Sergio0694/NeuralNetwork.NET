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
            double[,]
                m_gpu = Gpu.Default.Allocate(m),
                mresult_gpu = Gpu.Default.Allocate<double>(w, h);

            // Wrapper
            void Kernel(int ki)
            {
                for (int j = 0; j < w; j++)
                    mresult_gpu[j, ki] = m_gpu[ki, j];
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h, Kernel);

            // Return the results
            Gpu.Free(m_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
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
            double[,]
                m1_gpu = Gpu.Default.Allocate(m1),
                m2_gpu = Gpu.Default.Allocate(m2),
                mresult_gpu = Gpu.Default.Allocate<double>(l, w);   // l because the first matrix will be transposed

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
                    sum += m1_gpu[k, i] * m2_gpu[k, j];
                }
                mresult_gpu[i, j] = sum;
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, l * w, Kernel);

            // Free memory and copy the result back
            Gpu.Free(m1_gpu);
            Gpu.Free(m2_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
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
                double activation = ActivationFunctionProvider.Activation(sum);
                mresult_gpu[i, j] = activation;
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
                        zPrime = ActivationFunctionProvider.ActivationPrime(z_gpu[i, j]),
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
            double[,]
                z_gpu = Gpu.Default.Allocate(z),
                d_gpu = Gpu.Default.Allocate(delta);

            // Wrapper
            void Kernel(int i)
            {
                // Save the index and iterate for each column
                for (int j = 0; j < w; j++)
                {
                    z_gpu[i, j] = ActivationFunctionProvider.ActivationPrime(z_gpu[i, j]) * d_gpu[i, j];
                }
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h, Kernel);

            // Free memory and copy the result back
            Gpu.Free(d_gpu);
            Gpu.Copy(z_gpu, z);
            Gpu.Free(z_gpu);
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
            // Initialize the parameters and the result matrix
            int h = m.GetLength(0);
            int w = m.GetLength(1);
            double[,]
                m_gpu = Gpu.Default.Allocate(m),
                mresult_gpu = Gpu.Default.Allocate<double>(h, w);

            // Wrapper
            void Kernel(int ki)
            {
                // Apply the activation to the current row
                for (int j = 0; j < w; j++)
                {
                    mresult_gpu[ki, j] = ActivationFunctionProvider.Activation(m_gpu[ki, j]);
                }
            }

            // Execute the multiplication in parallel
            Gpu.Default.For(0, h, Kernel);

            // Free memory and copy the result back
            Gpu.Free(m_gpu);
            double[,] result = Gpu.CopyToHost(mresult_gpu);
            Gpu.Free(mresult_gpu);
            return result;
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
            double[,]
                m1_gpu = Gpu.Default.Allocate(m1),
                m2_gpu = Gpu.Default.Allocate(m2);
            double[] result_gpu = Gpu.Default.Allocate<double>(h);

            // Wrapper
            void Kernel(int i)
            {
                // Local sum over each row
                double row = 0;
                for (int j = 0; j < w; j++)
                {
                    double delta = m1_gpu[i, j] - m2_gpu[i, j];
                    row += delta * delta;
                }
                result_gpu[i] = row;
            }

            // Run the first kernel and compute the sum vector
            Gpu.Default.For(0, h, Kernel);
            Gpu.Free(m1_gpu);
            Gpu.Free(m2_gpu);

            // Aggregate the results in the sum vector
            double cost = Gpu.Default.Aggregate(result_gpu, (n1, n2) => n1 + n2);
            Gpu.Free(result_gpu);
            return cost / 2;
        }

        #endregion
    }
}
