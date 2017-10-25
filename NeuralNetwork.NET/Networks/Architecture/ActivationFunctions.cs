using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Networks.Architecture
{
    /// <summary>
    /// A static collection of available activation functions
    /// </summary>
    public static class ActivationFunctions
    {
        /// <summary>
        /// Applies the sigmoid function, 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Sigmoid(this double x) => 1 / (1 + Math.Exp(-x));

        /// <summary>
        /// Applies the sigmoid prime function, 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double SigmoidPrime(this double x)
        {
            double
                exp = Math.Exp(x),
                sum = 1 + exp,
                square = sum * sum,
                div = exp / square;
            return div;
        }

        /// <summary>
        /// Applies the tanh function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Tanh(this double x)
        {
            double e2x = Math.Exp(2 * x);
            return (e2x - 1) / (e2x + 1);
        }

        /// <summary>
        /// Applies the derivative of the <see cref="Tanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double TanhPrime(this double x)
        {
            double
                eminus2x = Math.Exp(-x),
                e2x = Math.Exp(x),
                sum = eminus2x + e2x,
                square = sum * sum,
                div = 4 / square;
            return div;
        }

        /// <summary>
        /// Applies the rectifier function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ReLU(this double x) => x > 0 ? x : 0;

        /// <summary>
        /// Applies the derivative of the <see cref="Tanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The real derivative is indetermined when x is 0</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ReLUPrime(this double x) => x <= 0 ? 0 : 1;

        /// <summary>
        /// Applies the softplus function, ln(1 + e^x)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The derivative of the softplus is the <see cref="Sigmoid"/> function</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Softplus(this double x)
        {
            double
                exp = Math.Exp(x),
                sum = 1 + exp,
                ln = Math.Log(sum);
            return ln;
        }
    }
}
