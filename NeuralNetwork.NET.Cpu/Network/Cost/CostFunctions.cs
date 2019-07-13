using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Activations.Delegates;

namespace NeuralNetworkDotNet.Network.Cost
{
    /// <summary>
    /// A collection of cost functions available for the neural networks
    /// </summary>
    public static class CostFunctions
    {
        #region Cost

        /// <summary>
        /// Calculates the quadratic cost for the given outputs and expected results
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static float QuadraticCost([NotNull] Tensor yHat, [NotNull] Tensor y)
        {
            Guard.IsTrue(yHat.Shape.N == y.Shape.N, "The input tensors don't have the same number of samples");
            Guard.IsTrue(yHat.Shape.CHW == y.Shape.CHW, "The input tensors don't have the same number of features per sample");

            int h = yHat.Shape.N, w = yHat.Shape.CHW;

            using (var v = Tensor.New(h, 1))
            {
                // Calculate the cost (half the squared difference)
                void Kernel(int i)
                {
                    var offset = i * w;
                    var sum = 0f;

                    ref var ryHat = ref yHat.Span.GetPinnableReference();
                    ref var ry = ref y.Span.GetPinnableReference();

                    // Compute the partial sum
                    for (var j = 0; j < w; j++)
                    {
                        var target = offset + j;
                        float
                            delta = Unsafe.Add(ref ryHat, target) - Unsafe.Add(ref ry, target),
                            square = delta * delta;
                        sum += square;
                    }

                    v.Span[i] = sum;
                }

                Parallel.For(0, h, Kernel);

                // Sum the partial costs
                var cost = 0f;
                ref var rv = ref v.Span.GetPinnableReference();
                for (var i = 0; i < h; i++) cost += Unsafe.Add(ref rv, i);

                return cost / 2;
            }
        }

        /// <summary>
        /// Calculates the cross-entropy cost for a given feedforward result
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static float CrossEntropyCost([NotNull] Tensor yHat, [NotNull] Tensor y)
        {
            Guard.IsTrue(yHat.Shape.N == y.Shape.N, "The input tensors don't have the same number of samples");
            Guard.IsTrue(yHat.Shape.CHW == y.Shape.CHW, "The input tensors don't have the same number of features per sample");

            int h = yHat.Shape.N, w = yHat.Shape.CHW;

            using (var v = Tensor.New(h, 1))
            {
                // Function to calculate cost for each sample
                void Kernel(int i)
                {
                    var offset = i * w;
                    var sum = 0f;

                    ref var ryHat = ref yHat.Span.GetPinnableReference();
                    ref var ry = ref y.Span.GetPinnableReference();

                    for (var j = 0; j < w; j++)
                    {
                        var target = offset + j;
                        float
                            yi = Unsafe.Add(ref ry, target),
                            yHati = Unsafe.Add(ref ryHat, target),
                            left = yi * (float)Math.Log(yHati),
                            right = (1 - yi) * (float)Math.Log(1 - yHati),
                            partial = left + right;

                        switch (partial)
                        {
                            case float.NegativeInfinity: sum += -float.MaxValue; break;
                            case float.NaN: break;
                            case float.PositiveInfinity: throw new InvalidOperationException("Error calculating the cross-entropy cost");
                            default: sum += partial; break;
                        }
                    }

                    v.Span[i] = sum;
                }

                Parallel.For(0, h, Kernel);

                // Sum the partial results and normalize
                var cost = 0f;
                ref var rv = ref v.Span.GetPinnableReference();
                for (var i = 0; i < h; i++) cost += Unsafe.Add(ref rv, i);

                return -cost / h;
            }
        }

        /// <summary>
        /// Calculates the log-likelyhood cost for the given outputs and expected results
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        [Pure]
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public static float LogLikelyhoodCost([NotNull] Tensor yHat, [NotNull] Tensor y)
        {
            Guard.IsTrue(yHat.Shape.N == y.Shape.N, "The input tensors don't have the same number of samples");
            Guard.IsTrue(yHat.Shape.CHW == y.Shape.CHW, "The input tensors don't have the same number of features per sample");

            int h = yHat.Shape.N, w = yHat.Shape.CHW;

            using (var v = Tensor.New(h, 1))
            {
                // Kernel to compute the partial sum
                void Kernel(int i)
                {
                    int
                        offset = i * w,
                        iy = y.Span.Slice(offset, w).Argmax();

                    v.Span[i] = -(float)Math.Log(yHat.Span[offset + iy]);
                }

                Parallel.For(0, h, Kernel);

                // Sum the partial costs
                var cost = 0f;
                ref var rv = ref v.Span.GetPinnableReference();
                for (var i = 0; i < h; i++) cost += Unsafe.Add(ref rv, i);

                return cost;
            }
        }

        #endregion

        #region Derivative

        /// <summary>
        /// Calculates the derivative of the quadratic cost function for the given outputs, expected results and activity
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        /// <param name="z">The activity on the last network layer</param>
        /// <param name="activationPrime">The activation pime function for the last network layer</param>
        /// <param name="dx">The backpropagated error</param>
        public static void QuadraticCostPrime([NotNull] Tensor yHat, [NotNull] Tensor y, [NotNull] Tensor z, [NotNull] ActivationFunction activationPrime, [NotNull] Tensor dx)
        {
            Guard.IsTrue(yHat.Shape.N == y.Shape.N, "The input tensors don't have the same number of samples");
            Guard.IsTrue(yHat.Shape.CHW == y.Shape.CHW, "The input tensors don't have the same number of features per sample");
            Guard.IsTrue(yHat.Shape.N == dx.Shape.N, "The input tensor and the result tensor don't have the same number of samples");
            Guard.IsTrue(yHat.Shape.CHW == dx.Shape.CHW, "The input tensor and the result tensor don't have the same number of features per sample");

            int h = yHat.Shape.N, w = yHat.Shape.CHW;

            // Calculate (yHat - y) * activation'(z)
            void Kernel(int i)
            {
                // Save the index and iterate for each column
                var offset = i * w;

                ref var ryHat = ref yHat.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();
                ref var rz = ref z.Span.GetPinnableReference();
                ref var rdx = ref dx.Span.GetPinnableReference();

                for (var j = 0; j < w; j++)
                {
                    var index = offset + j;
                    float
                        difference = Unsafe.Add(ref ryHat, index) - Unsafe.Add(ref ry, index),
                        zPrime = activationPrime(Unsafe.Add(ref rz, index)),
                        hProduct = difference * zPrime;

                    Unsafe.Add(ref rdx, index) = hProduct;
                }
            }

            Parallel.For(0, h, Kernel);
        }

        /// <summary>
        /// Calculates the derivative cross-entropy cost for a given feedforward result
        /// </summary>
        /// <param name="yHat">The current results</param>
        /// <param name="y">The expected results for the dataset</param>
        /// <param name="z">The activity on the last network layer</param>
        /// <param name="activationPrime">The activation pime function for the last network layer</param>
        /// <param name="dx">The backpropagated error</param>
        public static void CrossEntropyCostPrime([NotNull] Tensor yHat, [NotNull] Tensor y, [NotNull] Tensor z, [NotNull] ActivationFunction activationPrime, [NotNull] Tensor dx)
        {
            CpuBlas.Subtract(yHat, y, dx);
        }

        #endregion
    }
}
