using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.Core.Helpers;
using NeuralNetworkDotNet.Core.Structs;

namespace NeuralNetworkDotNet.Cpu.cpuDNN
{
    public static class CpuDnn
    {
        /// <summary>
        /// Executes the forward pass on a fully connected layer
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        /// <param name="w">The layer weights</param>
        /// <param name="b">The layer biases</param>
        /// <param name="y">The output <see cref="Tensor"/> for the current layer</param>
        public static void FullyConnectedForward([NotNull] Tensor x, [NotNull] Tensor w, [NotNull] Tensor b, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.C == 1 && x.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(w.C == 1 && w.H == 1, nameof(w), "The w tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x.W == w.N, "The input tensor shape doesn't match the shape of the given weights");
            Guard.IsTrue((b.N, b.W) == (1, w.W), nameof(b), "The shape of the input biases isn't valid");
            Guard.IsTrue((y.N, y.W) == (x.N, w.W), nameof(y), "The output tensor doesn't have the right shape");

            int
                h = x.N,
                l = x.W,
                k = w.W;

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
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")] // Tensors in parallel kernel
        public static void FullyConnectedBackwardData([NotNull] Tensor w, [NotNull] Tensor dy, [NotNull] Tensor dx)
        {
            Guard.IsTrue(w.C == 1 && w.H == 1, nameof(w), "The w tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dy.C == 1 && dy.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dx.C == 1 && dx.H == 1, nameof(dx), "The dx tensor doesn't represent a 2D matrix");
            Guard.IsTrue(w.W == dy.W, nameof(w), "The weights tensor doesn't have a valid shape");
            Guard.IsTrue((dx.N, dx.W) == (dy.N, w.N), nameof(dx), "The input tensor doesn't have the right shape");

            using (var wt = Tensor.New(w.W, w.N))
            {
                CpuBlas.Transpose(w, wt);

                int
                    h = dy.N,
                    l = dy.W,
                    k = wt.W;

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
        public static void FullyConnectedBackwardFilter([NotNull] Tensor x, [NotNull] Tensor dy, [NotNull] Tensor dw)
        {
            Guard.IsTrue(x.C == 1 && x.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dy.C == 1 && dy.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(dw.C == 1 && dw.H == 1, nameof(dw), "The dx tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x.N == dy.N, "The input tensor doesn't match the number of samples from the delta");

            using (var xt = Tensor.New(x.CHW, x.N))
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
        public static void FullyConnectedBackwardBias([NotNull] Tensor dy, [NotNull] Tensor db)
        {
            Guard.IsTrue(dy.C == 1 && dy.H == 1, nameof(dy), "The dy tensor doesn't represent a 2D matrix");
            Guard.IsTrue(db.Shape == (1, 1, 1, dy.W), "Invalid db tensor shape");

            int
                n = dy.N,
                l = dy.CHW;

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
