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
            if (!y.MatchShape(x)) throw new ArgumentException(nameof(y), "The input tensors must have the same shape");
            if (!dx.MatchShape(x)) throw new ArgumentException(nameof(y), "The output tensor must have the same shape as the input");
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
    }
}
