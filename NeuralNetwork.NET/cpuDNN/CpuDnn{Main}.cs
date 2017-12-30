using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.cpuDNN
{
    public static class CpuDnn
    {
        #region Activation

        public static unsafe void ActivationForward(in Tensor x, [NotNull] ActivationFunction f, in Tensor y)
        {
            // Setup
            int n = x.Entities, l = x.Length;
            if (!y.MatchShape(x)) throw new ArgumentException("The target tensor must have the same shape as the input");
            float* py = y, px = x;

            // Execute the activation in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int target = offset + j;
                    py[target] = f(px[target]);
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        public static unsafe void ActivationBackward(in Tensor x, in Tensor y, [NotNull] ActivationFunction f_, in Tensor dx)
        {
            // Check
            if (!y.MatchShape(x)) throw new ArgumentException("The input tensors must have the same shape", nameof(y));
            if (!dx.MatchShape(x)) throw new ArgumentException("The output tensor must have the same shape as the input", nameof(y));
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, py = y, pdx = dx;

            // Loop in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < l; j++)
                {
                    int target = offset + j;
                    pdx[target] = f_(px[target]) * py[target];
                }
            }
            Parallel.For(0, n, Kernel).AssertCompleted();
        }

        #endregion

        #region Fully connected

        public static unsafe void FullyConnectedForward(in Tensor x, in Tensor w, in Tensor b, in Tensor y)
        {
            // Initialize the parameters and the result matrix
            if (x.Length != w.Entities) throw new ArgumentOutOfRangeException("Invalid tensors shapes");
            if (!b.MatchShape(1, w.Length)) throw new ArgumentException("Invalid biases shape", nameof(b));
            if (!y.MatchShape(x.Entities, w.Length)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(y));
            int
                h = x.Entities,
                l = x.Length,
                k = w.Length;
            float* pdy = y, px = x, pw = w, pb = b;

            // Execute the multiplication in parallel
            void Kernel(int i)
            {
                int offset = i * l;
                for (int j = 0; j < k; j++)
                {
                    // Perform the multiplication
                    int start = j;
                    float res = 0;
                    for (int q = 0; q < l; q++, start += k)
                    {
                        res += px[offset + q] * pw[start];
                    }
                    pdy[i * k + j] = res + pb[j]; // Sum the input vector to each component
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
        }

        public static unsafe void FullyConnectedBackwardData(in Tensor x, in Tensor w, in Tensor dy, [NotNull] ActivationFunction f_, in Tensor dx)
        {
            if (w.Entities != x.Length) throw new ArgumentException("The weights tensor doesn't have a valid shape", nameof(w));
            if (!x.MatchShape(dy.Entities, w.Entities)) throw new ArgumentException("The input tensor doesn't have the right shape", nameof(x));
            if (!dx.MatchShape(x)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(dx));
            Tensor.New(w.Length, w.Entities, out Tensor wt);
            CpuBlas.Transpose(w, wt);

            // Initialize the parameters and the result matrix
            int 
                h = dy.Entities,
                l = dy.Length,
                k = wt.Length;
            float* pdx = dx, px = x, pdy = dy, pwt = wt;

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
                        res += pdy[i1 + q] * pwt[i2];
                    }

                    // res has now the matrix multiplication result for position [i, j]
                    int zIndex = i * k + j;
                    pdx[zIndex] = f_(px[zIndex]) * res;
                }
            }
            Parallel.For(0, h, Kernel).AssertCompleted();
            wt.Free();
        }

        public static void FullyConnectedBackwardFilter(in Tensor x, in Tensor dy, in Tensor dw)
        {
            if (x.Entities != dy.Entities) throw new ArgumentException("The input tensor doesn't match the number of samples from the delta", nameof(x));
            Tensor.New(x.Length, x.Entities, out Tensor xt);
            CpuBlas.Transpose(x, xt);
            CpuBlas.Multiply(xt, dy, dw);
            xt.Free();
        }

        public static unsafe void FullyConnectedBackwardBias(in Tensor dy, in Tensor db)
        {
            // Preliminary checks and declarations
            if (!db.MatchShape(1, dy.Length)) throw new ArgumentException("The output tensor doesn't have the right shape", nameof(db));
            int
                n = dy.Entities,
                l = dy.Length;
            float* pdy = dy, pdb = db;

            // Compress the matrix
            void Kernel(int j)
            {
                float sum = 0;
                for (int i = 0; i < n; i++)
                    sum += pdb[i * l + j];
                pdy[j] = sum;
            }
            Parallel.For(0, l, Kernel).AssertCompleted();
        }

        #endregion
    }
}
