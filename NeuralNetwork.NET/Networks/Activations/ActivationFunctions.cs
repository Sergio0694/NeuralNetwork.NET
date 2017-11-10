using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Networks.Activations
{
    /// <summary>
    /// A static collection of available activation functions
    /// </summary>
    internal static class ActivationFunctions
    {
        /// <summary>
        /// Applies the sigmoid function, 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(this float x) => 1 / (1 + (float)Math.Exp(-x));

        /// <summary>
        /// Applies the sigmoid prime function, 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidPrime(this float x)
        {
            float
                exp = (float)Math.Exp(x),
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
        public static float Tanh(this float x)
        {
            float e2x = (float)Math.Exp(2 * x);
            return (e2x - 1) / (e2x + 1);
        }

        /// <summary>
        /// Applies the derivative of the <see cref="Tanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float TanhPrime(this float x)
        {
            float
                eminus2x = (float)Math.Exp(-x),
                e2x = (float)Math.Exp(x),
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
        public static float ReLU(this float x) => x > 0 ? x : 0;

        /// <summary>
        /// Applies the derivative of the <see cref="Tanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The real derivative is indetermined when x is 0</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLUPrime(this float x) => x <= 0 ? 0 : 1;

        /// <summary>
        /// Applies the leaky ReLU function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLU(this float x) => x > 0 ? x : 0.01f * x;

        /// <summary>
        /// Applies the derivative of the <see cref="LeakyReLU"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The real derivative is indetermined when x is 0</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLUPrime(this float x) => x > 0 ? 1 : 0.01f;

        /// <summary>
        /// Applies the softplus function, ln(1 + e^x)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The derivative of the softplus is the <see cref="Sigmoid"/> function</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Softplus(this float x)
        {
            float
                exp = (float)Math.Exp(x),
                sum = 1 + exp,
                ln = (float)Math.Log(sum);
            return ln;
        }

        /// <summary>
        /// Applies the exponential linear unit function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELU(this float x) => x >= 0 ? x : (float)Math.Exp(x) - 1;

        /// <summary>
        /// Applies the derivative of the <see cref="ELU"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELUPrime(this float x) => x >= 0 ? 1 : (float)Math.Exp(x);
    }
}
