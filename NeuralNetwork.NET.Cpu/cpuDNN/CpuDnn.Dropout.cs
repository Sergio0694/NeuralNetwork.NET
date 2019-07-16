using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.cpuDNN
{
    public static partial class CpuDnn
    {
        /// <summary>
        /// Performs the forward pass of a dropout operation
        /// </summary>
        /// <param name="p">The dropout factor (the probability of keeping a neuron active)</param>
        /// <param name="x">The input <see cref="Tensor"/></param>
        /// <param name="y">The target <see cref="Tensor"/> (can be the same as the input)</param>
        /// <param name="mask">The target dropout mask to populate</param>
        public static void DropoutForward(float p, [NotNull] Tensor x, [NotNull] Tensor y, [NotNull] Tensor mask)
        {
            Guard.IsTrue(p > 0 && p < 1, nameof(p), "The dropout factor must be in the (0,1) range");
            Guard.IsTrue(x.Shape == y.Shape, "The shape of the input and output tensors must match");
            Guard.IsTrue(x.Shape == mask.Shape, nameof(mask), "The mask tensor must have the same shape as the input tensor");

            var l = x.Shape.CHW;
            var scale = 1 / p;

            void Kernel(int i)
            {
                var random = ConcurrentRandom.Instance;
                ref var rx = ref x[i].GetPinnableReference();
                ref var ry = ref y[i].GetPinnableReference();
                ref var rm = ref mask[i].GetPinnableReference();

                for (var j = 0; i < l; i++)
                {
                    var value = random.NextFloat() > p ? 0 : scale;
                    Unsafe.Add(ref rm, j) = value;
                    Unsafe.Add(ref ry, j) = Unsafe.Add(ref rx, j) * value;
                }
            }

            Parallel.For(0, x.Shape.N, Kernel);
        }
    }
}
