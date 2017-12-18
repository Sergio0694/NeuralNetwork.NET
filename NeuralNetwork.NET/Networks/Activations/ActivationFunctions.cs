using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;

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
        public static float Sigmoid(float x) => 1 / (1 + (float)Math.Exp(-x));

        /// <summary>
        /// Applies the sigmoid prime function, 1 / (1 + e^(-x))
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SigmoidPrime(float x)
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
        public static float Tanh(float x)
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
        public static float TanhPrime(float x)
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
        /// Applies the LeCun tanh function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeCunTanh(float x)
        {
            const float divX = 2f / 3;
            const float scale = 1.7159f;
            float e2x = (float)Math.Exp(2 * divX * x);
            return scale * (e2x - 1) / (e2x + 1);
        }

        /// <summary>
        /// Applies the derivative of the <see cref="LeCunTanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeCunTanhPrime(float x)
        {
            const float numerator = 4.57573f;
            float
                exp = 2 * x / 3,
                ePlus = (float)Math.Exp(exp),
                eMinus = (float)Math.Exp(-exp),
                sum = ePlus + eMinus,
                square = sum * sum;
            return numerator / square;
        }

        /// <summary>
        /// Applies the rectifier function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLU(float x) => x > 0 ? x : 0;

        /// <summary>
        /// Applies the derivative of the <see cref="Tanh"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The real derivative is indetermined when x is 0</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ReLUPrime(float x) => x <= 0 ? 0 : 1;

        /// <summary>
        /// Applies the leaky ReLU function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLU(float x) => x > 0 ? x : 0.01f * x;

        /// <summary>
        /// Applies the derivative of the <see cref="LeakyReLU"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The real derivative is indetermined when x is 0</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LeakyReLUPrime(float x) => x > 0 ? 1 : 0.01f;

        /// <summary>
        /// Applies the the numerator part of the softmax activation function, e^x
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The derivative is not available, as it doesn't appear in the derivative of the log-likelyhood cost function</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Softmax(float x) => (float)Math.Exp(x);

        /// <summary>
        /// Applies the softplus function, ln(1 + e^x)
        /// </summary>
        /// <param name="x">The input to process</param>
        /// <remarks>The derivative of the softplus is the <see cref="Sigmoid"/> function</remarks>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Softplus(float x)
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
        public static float ELU(float x) => x >= 0 ? x : (float)Math.Exp(x) - 1;

        /// <summary>
        /// Applies the derivative of the <see cref="ELU"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ELUPrime(float x) => x >= 0 ? 1 : (float)Math.Exp(x);

        /// <summary>
        /// Applies the absolute ReLU linear unit function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float AbsoluteReLU(float x) => x >= 0 ? x : -x;

        /// <summary>
        /// Applies the derivative of the <see cref="AbsoluteReLU"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float AbsoluteReLUPrime(float x) => x >= 0 ? 1 : -1;

        /// <summary>
        /// Applies the identity function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Identity(float x) => x;

        /// <summary>
        /// Applies the derivative of the <see cref="Identity"/> function
        /// </summary>
        /// <param name="x">The input to process</param>
        [PublicAPI]
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Identityprime(float x) => 1;
    }
}
