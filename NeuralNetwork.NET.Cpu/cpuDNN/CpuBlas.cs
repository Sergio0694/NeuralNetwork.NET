using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.cpuDNN
{
    /// <summary>
    /// A <see langword="class"/> that exposes static BLAS (Basic Linear Algebra Subprograms) methods working on <see cref="Tensor"/> instances
    /// </summary>
    public static class CpuBlas
    {
        /// <summary>
        /// Transposes the input <see cref="Tensor"/> instance
        /// </summary>
        /// <param name="x">The <see cref="Tensor"/> to transpose</param>
        /// <param name="y">The destination <see cref="Tensor"/> that will hold the results</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void Transpose([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.C == 1 && x.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x.N == y.CHW, "The output tensor doesn't have a valid CHW configuration");
            Guard.IsTrue(x.CHW == y.N, "The output tensor doesn't have a valid N configuration");

            int n = x.N, l = x.CHW;

            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                    Unsafe.Add(ref ry, j * n + i) = Unsafe.Add(ref rx, offset + j);
            }

            Parallel.For(0, n, Kernel);
        }

        /// <summary>
        /// Performs the multiplication between two <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/> to multiply</param>
        /// <param name="x2">The second <see cref="Tensor"/> to multiply</param>
        /// <param name="y">The resulting <see cref="Tensor"/> to hold the results</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void Multiply([NotNull] Tensor x1, [NotNull] Tensor x2, [NotNull] Tensor y)
        {
            Guard.IsTrue(x1.C == 1 && x1.H == 1, nameof(x1), "The x1 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x2.C == 1 && x2.H == 1, nameof(x2), "The x2 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x1.CHW == x2.N, "The size of the input tensors isn't valid");
            Guard.IsTrue(x1.N == y.N, nameof(y), "The result tensor doesn't have the right N parameter");
            Guard.IsTrue(x2.CHW == y.CHW, nameof(y), "The result tensor doesn't have the right CHW parameter");

            int
                n = x1.N,
                l = x1.CHW,
                k = x2.CHW;

            void Kernel(int i)
            {
                var i1 = i * l;
                ref var rx1 = ref x1.Span.GetPinnableReference();
                ref var rx2 = ref x2.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < k; j++)
                {
                    var i2 = j;
                    var res = 0f;
                    for (var q = 0; q < l; q++, i2 += k)
                    {
                        res += Unsafe.Add(ref rx1, i1 + q) * Unsafe.Add(ref rx2, i2);
                    }

                    Unsafe.Add(ref ry, i * k + j) = res;
                }
            }

            Parallel.For(0, n, Kernel);
        }

        /// <summary>
        /// Performs the elementwise multiplication (Hadamard product) product between two <see cref="Tensor"/> instances
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/> to multiply</param>
        /// <param name="x2">The second <see cref="Tensor"/> to multiply</param>
        /// <param name="y">The resulting <see cref="Tensor"/> to hold the results</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void MultiplyElementwise([NotNull] Tensor x1, [NotNull] Tensor x2, [NotNull] Tensor y)
        {
            Guard.IsTrue(x1.C == 1 && x1.H == 1, nameof(x1), "The x1 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x2.C == 1 && x2.H == 1, nameof(x2), "The x2 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue((x1.N, x1.CHW) == (x2.N, x2.CHW), "The x1 and x2 parameters don't have the same shape");
            Guard.IsTrue((x1.N, x1.CHW) == (y.N, y.CHW), nameof(y), "The y parameter don't have the same shape");

            int n = x1.N, l = x1.CHW;

            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx1 = ref x1.Span.GetPinnableReference();
                ref var rx2 = ref x2.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (int j = 0; j < l; j++)
                {
                    int position = offset + j;
                    Unsafe.Add(ref ry, position) = Unsafe.Add(ref rx1, position) * Unsafe.Add(ref rx2, position);
                }
            }

            Parallel.For(0, n, Kernel);
        }

        /// <summary>
        /// Sums an input <see cref="Tensor"/> into a target <see cref="Tensor"/> instance
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to sum</param>
        /// <param name="y">The output <see cref="Tensor"/> that will hold the results</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        public static void Sum([NotNull] Tensor x, [NotNull] Tensor y)
        {
            Guard.IsTrue(x.C == 1 && x.H == 1, nameof(x), "The x tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue((x.N, x.CHW) == (y.N, y.CHW), "The x and y parameters don't have the same shape");

            int n = y.N, l = y.CHW;

            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx = ref x.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    var position = offset + j;
                    Unsafe.Add(ref ry, position) += Unsafe.Add(ref rx, position);
                }

            }

            Parallel.For(0, n, Kernel);
        }

        /// <summary>
        /// Subtracts two <see cref="Tensor"/> instances, element wise
        /// </summary>
        /// <param name="x1">The first <see cref="Tensor"/></param>
        /// <param name="x2">The second <see cref="Tensor"/></param>
        /// <param name="y">The resulting <see cref="Tensor"/> - it can be the same as one of the inputs</param>
        /// <exception cref="System.ArgumentException">The size of one of the input <see cref="Tensor"/> instances isn't valid</exception>
        internal static void Subtract([NotNull] Tensor x1, [NotNull] Tensor x2, [NotNull] Tensor y)
        {
            Guard.IsTrue(x1.C == 1 && x1.H == 1, nameof(x1), "The x1 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(x2.C == 1 && x2.H == 1, nameof(x2), "The x2 tensor doesn't represent a 2D matrix");
            Guard.IsTrue(y.C == 1 && y.H == 1, nameof(y), "The y tensor doesn't represent a 2D matrix");
            Guard.IsTrue((x1.N, x1.CHW) == (x2.N, x2.CHW), "The x1 and x2 parameters don't have the same shape");
            Guard.IsTrue((x1.N, x1.CHW) == (y.N, y.CHW), nameof(y), "The y parameter don't have the same shape");

            int n = x1.N, l = x1.CHW;

            void Kernel(int i)
            {
                var offset = i * l;
                ref var rx1 = ref x1.Span.GetPinnableReference();
                ref var rx2 = ref x2.Span.GetPinnableReference();
                ref var ry = ref y.Span.GetPinnableReference();

                for (var j = 0; j < l; j++)
                {
                    int position = offset + j;
                    Unsafe.Add(ref ry, position) = Unsafe.Add(ref rx1, position) - Unsafe.Add(ref rx2, position);
                }
            }

            Parallel.For(0, n, Kernel);
        }
    }
}
