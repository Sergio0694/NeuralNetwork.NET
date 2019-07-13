using System.Diagnostics.CodeAnalysis;
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
        /// Executes the forward pass on a fully connected layer
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        /// <param name="w">The layer weights</param>
        /// <param name="b">The layer biases</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void FullyConnectedForward([NotNull] Tensor x, [NotNull] Tensor w, [NotNull] Tensor b, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.Shape.C == 1 && x.Shape.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(w.Shape.C == 1 && w.Shape.H == 1, nameof(w), "The w tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.Shape.C == 1 && y.Shape.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x.Shape.W == w.Shape.N, "The input tensor shape doesn't match the shape of the given weights");
            Guard.IsTrue((b.Shape.N, b.Shape.W) == (1, w.Shape.W), nameof(b), "The shape of the input biases isn't valid");
            Guard.IsTrue((y.Shape.N, y.Shape.W) == (x.Shape.N, w.Shape.W), nameof(y), "The output tensor doesn't have the right shape");

            int
                h = x.Shape.N,
                l = x.Shape.W,
                k = w.Shape.W;

            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx = ref x.Span.GetPinnableReference();
                ref var rw = ref w.Span.GetPinnableReference();
                ref var rb = ref b.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < k; j++)
                {
                    var start = j;
                    var res = 0f;

                    for (var q = 0; q < l; q++, start += k)
                    {
                        res += Unsafe.Add(ref rx, offset + q) * Unsafe.Add(ref rw, start);
                    }

                    Unsafe.Add(ref ry, i * k + j) = res + Unsafe.Add(ref rb, j); // Multiplication result + bias
                }
            }

            Parallel.For(0, h, Kernel);
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer
        /// </summary>
        /// <param name="w">The layer weights</param>
        /// <param name="dy">The output error delta</param>
        /// <param name="dx">The resulting input error delta</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")] // Tensors in parallel kernel
        public static void FullyConnectedBackwardData([NotNull] Tensor w, [NotNull] Tensor dy, [NotNull] Tensor dx)
        {
            Guard.IsTrue(w.Shape.C == 1 && w.Shape.H == 1, nameof(w), "The w tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dy.Shape.C == 1 && dy.Shape.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dx.Shape.C == 1 && dx.Shape.H == 1, nameof(dx), "The dx tensor doesn't represent a 2D matrix");
            Guard.IsTrue(w.Shape.W == dy.Shape.W, nameof(w), "The weights tensor doesn't have a valid shape");
            Guard.IsTrue((dx.Shape.N, dx.Shape.W) == (dy.Shape.N, w.Shape.N), nameof(dx), "The input tensor doesn't have the right shape");

            using (var wt = Tensor.New(w.Shape.W, w.Shape.N))
            {
                CpuBlas.Transpose(w, wt);

                int
                    h = dy.Shape.N,
                    l = dy.Shape.W,
                    k = wt.Shape.W;

                void Kernel(int i)
                {
                    var i1 = i * l;
                    ref var rwt = ref wt.Span.GetPinnableReference();
                    ref var rdy = ref dy.Span.GetPinnableReference();
                    ref var rdx = ref dx.Span.GetPinnableReference();

                    for (var j = 0; j < k; j++)
                    {
                        var i2 = j;
                        var res = 0f;
                        for (var q = 0; q < l; q++, i2 += k)
                        {
                            res += Unsafe.Add(ref rdy, i1 + q) * Unsafe.Add(ref rwt, i2);
                        }

                        Unsafe.Add(ref rdx, i * k + j) = res;
                    }
                }

                Parallel.For(0, h, Kernel);
            }
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer to calculate the gradient with respect to the weights
        /// </summary>
        /// <param name="x">The layer inputs</param>
        /// <param name="dy">The layer output error delta</param>
        /// <param name="dw">The resulting weights gradient <see cref="Tensor"/></param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void FullyConnectedBackwardFilter([NotNull] Tensor x, [NotNull] Tensor dy, [NotNull] Tensor dw)
        {
            Guard.IsTrue(x.Shape.C == 1 && x.Shape.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dy.Shape.C == 1 && dy.Shape.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dw.Shape.C == 1 && dw.Shape.H == 1, nameof(dw), "The dx tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x.Shape.N == dy.Shape.N, "The input tensor doesn't match the number of samples from the delta");

            using (var xt = Tensor.New(x.Shape.CHW, x.Shape.N))
            {
                CpuBlas.Transpose(x, xt);
                CpuBlas.Multiply(xt, dy, dw);
            }
        }

        /// <summary>
        /// Executes the backward pass on a fully connected layer to calculate the gradient with respect to the biases
        /// </summary>
        /// <param name="dy">The layer output error delta</param>
        /// <param name="db">The resulting biases gradient <see cref="Tensor"/></param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void FullyConnectedBackwardBias([NotNull] Tensor dy, [NotNull] Tensor db)
        {
            Guard.IsTrue(dy.Shape.C == 1 && dy.Shape.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(db.Shape == (1, 1, 1, dy.Shape.W), "Invalid db tensor shape");

            int
                n = dy.Shape.N,
                l = dy.Shape.CHW;

            // Compress the tensor
            void Kernel(int j)
            {
                var sum = 0f;
                ref var rdy = ref dy.Span.GetPinnableReference();
                ref var rdb = ref db.Span.GetPinnableReference();

                for (var i = 0; i < n; i++)
                    sum += Unsafe.Add(ref rdy, i * l + j);
                Unsafe.Add(ref rdb, j) = sum;
            }

            Parallel.For(0, l, Kernel);
        }
    }
}
