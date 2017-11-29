using System;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A thread-safe <see cref="Random"/> wrapper class with some extension methods
    /// </summary>
    public static class ThreadSafeRandom
    {
        // Shared seed
        private static int _Seed = Environment.TickCount;

        // Random instance that provides the pseudo-random numbers
        private static readonly ThreadLocal<Random> RandomInstance = new ThreadLocal<Random>(() => new Random(Interlocked.Increment(ref _Seed)));

        #region Base extensions

        /// <summary>
        /// Returns the next uniform value with the desired range
        /// </summary>
        /// <param name="scale">The desired scale for the uniform distribution</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextUniform(float scale = 1)
        {
            return (float)RandomInstance.Value.NextDouble() * scale * (NextBool() ? 1f : -1f);
        }

        /// <summary>
        /// Returns the next gaussian random value (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="deviation">The standard deviation for the distribution</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextGaussian(float deviation = 1)
        {
            double u1 = 1.0 - RandomInstance.Value.NextDouble(), u2 = 1.0 - RandomInstance.Value.NextDouble();
            return (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2) * deviation;
        }

        /// <summary>
        /// Returns the next <see cref="int"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int NextInt(int min = 0, int max = int.MaxValue) => RandomInstance.Value.Next(min, max);

        /// <summary>
        /// Returns the next <see cref="float"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float NextFloat() => (float)RandomInstance.Value.NextDouble();

        /// <summary>
        /// Returns the next <see cref="bool"/> random value
        /// </summary>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool NextBool() => RandomInstance.Value.NextDouble() >= 0.5;

        #endregion

        #region Matrix generators

        /// <summary>
        /// Returns a new matrix filled with values from a distribution ~N(0, deviation)
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="deviation">The deviation of the Gaussian distribution</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextGaussianMatrix(int x, int y, float deviation = 1) => NextMatrix(x, y, () => NextGaussian(deviation));

        /// <summary>
        /// Returns a new matrix filled with values from a distribution ~U(-scale, scale)
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="scale">The scale of the uniform distribution</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextUniformMatrix(int x, int y, float scale = 0.05f) => NextMatrix(x, y, () => NextUniform(scale));

        // Matrix initialization
        [Pure, NotNull]
        private static float[,] NextMatrix(int x, int y, Func<float> provider)
        {
            // Checks
            if (x <= 0 || y <= 0) throw new ArgumentOutOfRangeException("The size of the matrix isn't valid");

            // Prepare the local variables
            float[,] result = new float[x, y];

            // Populate the matrix in parallel
            Parallel.For(0, x, i =>
            {
                int offset = i * y;
                unsafe
                {
                    // Iterate over each row
                    fixed (float* r = result)
                        for (int j = 0; j < y; j++)
                            r[offset + j] = provider();
                }
            }).AssertCompleted();
            return result;
        }

        #endregion

        #region Keras weights initialization

        /// <summary>
        /// Returns a new matrix filled with values from the LeCun uniform distribution
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="fanIn">The input neurons</param>
        /// <remarks>LeCun 98, Efficient Backprop, <see cref="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf"/></remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextLeCunUniformMatrix(int x, int y, int fanIn = 0)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanIn == 0) fanIn = x;
            float scale = (float)Math.Sqrt(3f / fanIn);
            return NextUniformMatrix(x, y, scale);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the Glorot & Bengio normal distribution
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        /// <remarks>See Glorot & Bengio, AISTATS 2010</remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextGlorotNormalMatrix(int x, int y, int fanIn = 0, int fanOut = 0)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanIn == 0) fanIn = x;
            if (fanOut < 0) throw new ArgumentOutOfRangeException(nameof(fanOut), "The fan out must be a positive number");
            if (fanOut == 0) fanOut = x;
            float scale = (float)Math.Sqrt(2f / (fanIn + fanOut));
            return NextGaussianMatrix(x, y, scale);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the Glorot & Bengio uniform distribution
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="fanIn">The input neurons</param>
        /// <param name="fanOut">The output neurons</param>
        /// <remarks>See Glorot & Bengio, AISTATS 2010, <see cref="http://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py"/></remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextGlorotUniformMatrix(int x, int y, int fanIn = 0, int fanOut = 0)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanIn == 0) fanIn = x;
            if (fanOut < 0) throw new ArgumentOutOfRangeException(nameof(fanOut), "The fan out must be a positive number");
            if (fanOut == 0) fanOut = x;
            float scale = (float)Math.Sqrt(6f / (fanIn + fanOut));
            return NextUniformMatrix(x, y, scale);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the He et al. normal distribution
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="fanIn">The input neurons</param>
        /// <remarks>See He et al., <see cref="http://arxiv.org/abs/1502.01852"/></remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextHeEtAlNormalMatrix(int x, int y, int fanIn = 0)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanIn == 0) fanIn = x;
            float scale = (float)Math.Sqrt(2f / fanIn);
            return NextGaussianMatrix(x, y, scale);
        }

        /// <summary>
        /// Returns a new matrix filled with values from the He et al. uniform distribution
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="fanIn">The input neurons</param>
        /// <remarks>See He et al., <see cref="http://arxiv.org/abs/1502.01852"/></remarks>
        [PublicAPI]
        [Pure, NotNull]
        public static float[,] NextHeEtAlUniformMatrix(int x, int y, int fanIn = 0)
        {
            if (fanIn < 0) throw new ArgumentOutOfRangeException(nameof(fanIn), "The fan in must be a positive number");
            if (fanIn == 0) fanIn = x;
            float scale = (float)Math.Sqrt(6f / fanIn);
            return NextUniformMatrix(x, y, scale);
        }

        #endregion

        #region Misc

        /// <summary>
        /// Creates a dropout mask with the given size and probability
        /// </summary>
        /// <param name="x">The height of the matrix</param>
        /// <param name="y">The width of the matrix</param>
        /// <param name="dropout">The dropout probability (indicates the probability of keeping a neuron active)</param>
        [Pure, NotNull]
        public static float[,] NextDropoutMask(int x, int y, float dropout)
        {
            float scale = 1 / dropout;
            return NextMatrix(x, y, () => NextFloat() > dropout ? 0 : scale);
        }

        /// <summary>
        /// Returns a new vector filled with values from the Gaussian distribution (mean 0, standard deviation 1)
        /// </summary>
        /// <param name="n">The length of the vector</param>
        [PublicAPI]
        [Pure, NotNull]
        public static float[] NextGaussianVector(int n)
        {
            float[] v = new float[n];
            unsafe
            {
                fixed (float* pv = v)
                    for (int i = 0; i < n; i++)
                        pv[i] = NextGaussian();
            }
            return v;
        }

        #endregion
    }
}
