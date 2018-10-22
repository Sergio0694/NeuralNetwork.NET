using System;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    /// <summary>
    /// A class that exposes static BLAS (Basic Linear Algebra Subprograms) methods working on <see cref="Tensor"/> instances
    /// </summary>
    public static class CpuBlas
    {
        /// <summary>
        /// Transposes the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="x">The <see cref="Tensor"/> to transpose</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void Transpose(in Tensor x, in Tensor y)
        {
            // Setup
            if (!y.MatchShape(x.Length, x.Entities)) throw new ArgumentException("The output tensor doesn't have the right shape");
            int n = x.Entities, l = x.Length;
            float* px = x, py = y;

            // Execute the transposition in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                    py[j * n + i] = px[offset + j];
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the multiplication between two matrices
        /// </summary>
        /// <param name="x1">The first matrix to multiply</param>
        /// <param name="x2">The second matrix to multiply</param>
        /// <param name="y">The resulting matrix</param>
        public static unsafe void Multiply(in Tensor x1, in Tensor x2, in Tensor y)
        {
            // Initialize the parameters and the result matrix
            if (x1.Length != x2.Entities) throw new ArgumentOutOfRangeException(nameof(x1), "Invalid matrices sizes");
            if (!y.MatchShape(x1.Entities, x2.Length)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(y));
            int
                n = x1.Entities,
                l = x1.Length,
                k = x2.Length;
            float* px1 = x1, px2 = x2, py = y;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
                int i1 = i * l;
                for (int j = 0; j < k; j++)
                {
                    // Perform the multiplication
                    int i2 = j;
                    float res = 0;
                    for (int q = 0; q < l; q++, i2 += k)
                    {
                        res += px1[i1 + q] * px2[i2];
                    }
                    py[i * k + j] = res;
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Performs the elementwise multiplication (Hadamard product) product between two <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/></param>
        /// <param name="x2">The second <see cref="Tensor"/></param>
        /// <param name="y">The resulting <see cref="Tensor"/></param>
        public static unsafe void MultiplyElementwise(in Tensor x1, in Tensor x2, in Tensor y)
        {
            // Check
            int
                n = x1.Entities,
                l = x1.Length;
            if (!x1.MatchShape(x2)) throw new ArgumentException("The two input tensors must be of equal shape");
            if (!x1.MatchShape(y)) throw new ArgumentException("The output tensor must have the same shape as the input tensors", nameof(y));
            float* px1 = x1, px2 = x2, py = y;

            // Loop in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int position = offset + j;
                    py[position] = px1[position] * px2[position];
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Sums a series of input <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="inputs">A <see cref="Span{T}"/> containing the input <see cref="Tensor"/> instances to sum</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void Sum(Span<Tensor> inputs, in Tensor y)
        {
            if (inputs.Length == 0) throw new ArgumentException("The inputs can't be empty", nameof(inputs));
            int
                count = inputs.Length,
                n = y.Entities,
                l = y.Length;
            fixed (Tensor * p = inputs)
            {
                // Initial checks
                float** ps = stackalloc float*[count];
                for (int i = 0; i < count; i++)
                {
                    if (p[i].Entities != n || p[i].Length != l)
                        throw new ArgumentException("The input tensors must have the same size as the output tensor", nameof(inputs));
                    ps[i] = p[i];
                }

                // Sum the tensors in parallel
                float* py = y;
                void Kernel(int i)
                {
                    int offset = i * l;
                    for (int j = 0; j < l; j++)
                    {
                        int target = offset + j;
                        py[target] = 0;
                        for (int z = 0; z < count; z++)
                            py[target] += ps[z][target];
                    }

                }
                Parallel.For(0, n, Kernel).AssertCompleted();
            }
        }

        /// <summary>
        /// Subtracts two <see cref="Tensor"/> instances, element wise
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/></param>
        /// <param name="x2">The second <see cref="Tensor"/></param>
        /// <param name="y">The resulting <see cref="Tensor"/> - it can be the same as one of the inputs</param>
        internal static unsafe void Subtract(in Tensor x1, in Tensor x2, in Tensor y)
        {
            int
                n = x1.Entities,
                l = x1.Length;
            if (!x1.MatchShape(x2)) throw new ArgumentException("The two input tensors must be of equal shape");
            if (!x1.MatchShape(y)) throw new ArgumentException("The output tensor must have the same shape as the input tensors", nameof(y));

            // Subtract in parallel
            float* px1 = x1, px2 = x2, py = y;
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int position = offset + j;
                    py[position] = px1[position] - px2[position];
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }
    }
}
