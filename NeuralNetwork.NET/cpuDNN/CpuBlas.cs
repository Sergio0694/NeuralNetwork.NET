using System;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.cpuDNN
{
    public static class CpuBlas
    {
        /// <summary>
        /// Transposes the input <see cref="Tensor"/>
        /// </summary>
        /// <param name="x">The <see cref="Tensor"/> to transpose</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static unsafe void Transpose(in this Tensor x, in Tensor y)
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
        /// <param name="m1">The first matrix to multiply</param>
        /// <param name="m2">The second matrix to multiply</param>
        /// <param name="result">The resulting matrix</param>
        public static unsafe void Multiply(in this Tensor x1, in Tensor x2, in Tensor y)
        {
            // Initialize the parameters and the result matrix
            if (x1.Length != x2.Entities) throw new ArgumentOutOfRangeException("Invalid matrices sizes");
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
    }
}
