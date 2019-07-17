using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.cpuDNN
{
    public static partial class CpuDnn
    {
        /// <summary>
        /// Executes the forward pass on a depth stacking layer
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/> instance to stack</param>
        /// <param name="x2">The second <see cref="Tensor"/> instance to stack</param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static void DepthConcatenationForward([NotNull] Tensor x1, [NotNull] Tensor x2, [NotNull] Tensor y)
        {
            Guard.IsFalse(x1.Shape.N == 0, nameof(x1), "The first input can't be empty");
            Guard.IsFalse(x2.Shape.N == 0, nameof(x2), "The second input tensor can't be empty");
            Guard.IsTrue(x1.Shape.N == x2.Shape.N, "The input tensors must have the same number of samples");
            Guard.IsTrue((x1.Shape.H, x1.Shape.W) == (x2.Shape.H, x2.Shape.W), "The input tensors don't have a matching shape");
            Guard.IsTrue(x1.Shape.NCHW + x1.Shape.NCHW == y.Shape.NCHW, nameof(y), "The output tensor doesn't have the right size");
            Guard.IsTrue(x1.Shape.N == y.Shape.N, nameof(y), "The output tensor must have the same number of samples as the inputs");

            // Concatenate the tensors in parallel
            void Kernel(int i)
            {
                x1[i].CopyTo(y[i].Slice(0, x1.Shape.CHW));
                x2[i].CopyTo(y[i].Slice(x1.Shape.CHW, x2.Shape.CHW));
            }

            Parallel.For(0, x1.Shape.N, Kernel);
        }

        /// <summary>
        /// Executes the backward pass on a depth stacking layer
        /// </summary>
        /// <param name="dy">The input <see cref="Tensor"/> with the error delta to backpropagate</param>
        /// <param name="dx1">The first delta <see cref="Tensor"/></param>
        /// <param name="dx2">The second delta <see cref="Tensor"/></param>
        public static void DepthConcatenationBackward([NotNull] Tensor dy, [NotNull] Tensor dx1, [NotNull] Tensor dx2)
        {
            Guard.IsFalse(dx1.Shape.N == 0, nameof(dx1), "The first delta tensor can't be empty");
            Guard.IsFalse(dx2.Shape.N == 0, nameof(dx2), "The second delta tensor can't be empty");
            Guard.IsTrue(dx1.Shape.N == dx2.Shape.N, "The delta tensors must have the same number of samples");
            Guard.IsTrue((dx1.Shape.H, dx1.Shape.W) == (dx2.Shape.H, dx2.Shape.W), "The delta tensors don't have a matching shape");
            Guard.IsTrue(dx1.Shape.NCHW + dx1.Shape.NCHW == dy.Shape.NCHW, nameof(dy), "The input delta tensor doesn't have the right size");
            Guard.IsTrue(dx1.Shape.N == dy.Shape.N, nameof(dy), "The input delta tensor must have the same number of samples as the inputs");

            // Backpropagate in parallel
            void Kernel(int i)
            {
                dy[i].Slice(dx1.Shape.CHW).CopyTo(dx1[i]);
                dy[i].Slice(dx1.Shape.CHW, dx2.Shape.CHW).CopyTo(dx2[i]);
            }

            Parallel.For(0, dy.Shape.N, Kernel);
        }
    }
}
