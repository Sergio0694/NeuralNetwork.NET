using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Activations;
using NeuralNetworkDotNet.Network.Activations.Delegates;

namespace NeuralNetworkDotNet.cpuDNN
{
    /// <summary>
    /// A <see langword="class"/> that contains primitives to implement a neural network running on CPU
    /// </summary>
    public static partial class CpuDnn
    {
        /// <summary>
        /// Executes the input activation function on the target <see cref="Tensor"/>
        /// </summary>
        /// <param name="x">The layer input <see cref="Tensor"/></param>
        /// <param name="f">The activation function to apply to the input</param>
        /// <param name="y">The output <see cref="Tensor"/>, it can be the same as the input</param>
        public static void ActivationForward([NotNull] Tensor x, [NotNull] ActivationFunction f, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape == y.Shape, "The target tensor must have the same shape as the input");

            int n = x.Shape.N, l = x.Shape.CHW;

            // Execute the activation in parallel
            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var target = offset + j;
                    Unsafe.Add(ref ry, target) = f(Unsafe.Add(ref rx, target));
                }
            }

            Parallel.For(0, n, Kernel);
        }

        /// <summary>
        /// Performs the softmax activation on the input <see cref="Tensor"/> and applies the output normalization
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/></param>
        /// <param name="y">The output <see cref="Tensor"/></param>
        public static void SoftmaxForward([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape == y.Shape, "The target tensor must have the same shape as the input");

            int n = x.Shape.N, l = x.Shape.CHW;

            // Activation
            void ActivationWithAggregate(int i)
            {
                var offset = i * l;
                var sum = 0f;

                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var target = offset + j;
                    var value = ActivationFunctions.Softmax(Unsafe.Add(ref rx, target));

                    Unsafe.Add(ref ry, target) = value;
                    sum += value;
                }

                for (var j = 0; j < l; j++)
                    Unsafe.Add(ref ry, offset + j) /= sum;
            }

            Parallel.For(0, n, ActivationWithAggregate);
        }

        /// <summary>
        /// Executes the backward activation function on the target <see cref="Tensor"/>, with the given error delta
        /// </summary>
        /// <param name="y">The activity computed in the forwaard pass</param>
        /// <param name="dy">The current error delta to backpropagate</param>
        /// <param name="f">The derivative of the activation function used in the forward pass</param>
        /// <param name="dx">The resulting input error delta, it can be the same as the input <see cref="Tensor"/></param>
        public static void ActivationBackward([NotNull] Tensor y, [NotNull] Tensor dy, [NotNull] ActivationFunction f, [NotNull] Tensor dx)
        {
            Guard.IsTrue(dy.Shape == y.Shape, "The input tensors must have the same shape");
            Guard.IsTrue(dx.Shape == y.Shape, "The output tensor must have the same shape as the input");

            int n = y.Shape.N, l = y.Shape.CHW;

            // Activation prime in parallel
            void Kernel(int i)
            {
                var offset = i * l;
                ref var ry = ref y.Span.GetPinnableReference();
                ref var rdy = ref dy.Span.GetPinnableReference();
                ref var rdx = ref dx.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var target = offset + j;
                    Unsafe.Add(ref rdx, target) = f(Unsafe.Add(ref ry, target)) * Unsafe.Add(ref rdy, target);
                }
            }

            Parallel.For(0, n, Kernel);
        }
    }
}
