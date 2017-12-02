using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers.Delegates;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;

#pragma warning disable 1574

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class that forwards requests to execute matrix operations to the correct implementations
    /// </summary>
    internal static class MatrixServiceProvider
    {
        #region Initialization

        /// <summary>
        /// Assigns the delegates that will overwrite the default behavior of the service provider
        /// </summary>
        public static void SetupInjections(
            [NotNull] Func<float[,], float[,], float[], float[,]> multiplyWithSum,
            [NotNull] Func<float[,], float[,], float[,]> transposeMultiply,
            [NotNull] Func<float[,], ActivationFunction, float[,]> activation,
            [NotNull] Action<float[,], float[,], float[,], ActivationFunction> multiplyAndInPlaceActivationPrimeHadamard,
            [NotNull] ForwardConvolution convoluteForward,
            [NotNull] BackwardsConvolution convoluteBackwards,
            [NotNull] GradientConvolution convoluteGradient)
        {
            _MultiplyWithSumOverride = multiplyWithSum;
            _TransposeAndMultiplyOverride = transposeMultiply;
            _ActivationOverride = activation;
            _InPlaceMultiplyAndHadamardProductWithAcrivationPrime = multiplyAndInPlaceActivationPrimeHadamard;
            _ConvoluteForwardOverride = convoluteForward;
            _ConvoluteBackwardsOverride = convoluteBackwards;
            _ConvoluteGradientOverride = convoluteGradient;
        }

        /// <summary>
        /// Resets the previous injections and restores the default behavior of the service provider
        /// </summary>
        public static void ResetInjections()
        {
            _TransposeAndMultiplyOverride = null;
            _ActivationOverride = null;
            _InPlaceMultiplyAndHadamardProductWithAcrivationPrime = null;
            _ConvoluteForwardOverride = null;
            _ConvoluteBackwardsOverride = null;
            _ConvoluteGradientOverride = null;
        }

        #endregion

        #region Functional

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, TResult}"/> that multiplies two matrices and sums the input vector
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], float[], float[,]> _MultiplyWithSumOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSum"/> method
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyWithSum(in FloatSpan2D m1, float[,] m2, float[] v, out FloatSpan2D result)
        {
            m1.MultiplyWithSum(m2, v, out result);
            //return _MultiplyWithSumOverride?.Invoke(m1, m2, v) ?? m1.MultiplyWithSum(m2, v);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that transposes the first matrix and then multiplies it with the second one
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], float[,]> _TransposeAndMultiplyOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Transpose"/> and <see cref="MatrixExtensions.Multiply"/> methods in sequence
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TransposeAndMultiply(in FloatSpan2D m1, in FloatSpan2D m2, out FloatSpan2D result)
        {
            m1.Transpose(out FloatSpan2D m1t);
            m1t.Multiply(m2, out result);
            m1t.Free();
            //return _TransposeAndMultiplyOverride?.Invoke(m1, m2) ?? m1.Transpose().Multiply(m2);
        }

        /// <summary>
        /// A <see cref="Func{T, TResult}"/> that applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<float[,], ActivationFunction, float[,]> _ActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Activation"/> method
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Activation(in FloatSpan2D m, [NotNull] ActivationFunction activation, out FloatSpan2D result)
        {
            m.Activation(activation, out result);
            //return _ActivationOverride?.Invoke(m, activation) ?? m.Activation(activation);
        }

        #endregion

        #region Convolution

        /// <summary>
        /// A <see cref="ForwardConvolution"/> delegate that performs the forward convolution
        /// </summary>
        [CanBeNull]
        private static ForwardConvolution _ConvoluteForwardOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteForward"/> method
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ConvoluteForward(in FloatSpan2D m1, in VolumeInformation m1Info, [NotNull] float[,] m2, in VolumeInformation m2Info, [NotNull] float[] biases, out FloatSpan2D result)
        {
            if (_ConvoluteForwardOverride == null) m1.ConvoluteForward(m1Info, m2, m2Info, biases, out result);
            else _ConvoluteForwardOverride(m1, m1Info, m2, m2Info, biases, out result);
        }

        /// <summary>
        /// A <see cref="BackwardsConvolution"/> dekegate that performs the backwards convolution
        /// </summary>
        [CanBeNull]
        private static BackwardsConvolution _ConvoluteBackwardsOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteBackwards"/> method
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ConvoluteBackwards(in FloatSpan2D m1, in VolumeInformation m1Info, in FloatSpan2D m2, in VolumeInformation m2Info, out FloatSpan2D result)
        {
            if (_ConvoluteBackwardsOverride == null) m1.ConvoluteBackwards(m1Info, m2, m2Info, out result);
            else _ConvoluteBackwardsOverride(m1, m1Info, m2, m2Info, out result);
        }

        /// <summary>
        /// A <see cref="GradientConvolution"/> function that performs the gradient convolution
        /// </summary>
        [CanBeNull]
        private static GradientConvolution _ConvoluteGradientOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteGradient"/> method
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ConvoluteGradient(in FloatSpan2D m1, in VolumeInformation m1Info, in FloatSpan2D m2, in VolumeInformation m2Info, out FloatSpan2D result)
        {
            if (_ConvoluteGradientOverride == null) m1.ConvoluteGradient(m1Info, m2, m2Info, out result);
            else _ConvoluteGradientOverride(m1, m1Info, m2, m2Info, out result);
        }

        #endregion

        #region Side effect

        /// <summary>
        /// An <see cref="Action{T1, T2, T3}"/> that performs the activation prime function and then the Hadamard product with a matrix product
        /// </summary>
        [CanBeNull]
        private static Action<float[,], float[,], float[,], ActivationFunction> _InPlaceMultiplyAndHadamardProductWithAcrivationPrime;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceMultiplyAndHadamardProductWithAcrivationPrime"/> method
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void InPlaceMultiplyAndHadamardProductWithActivationPrime(in FloatSpan2D m, in FloatSpan2D di, in FloatSpan2D wt, [NotNull] ActivationFunction prime)
        {
            m.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(di, wt, prime);
           // if (_InPlaceMultiplyAndHadamardProductWithAcrivationPrime == null) m.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(di, wt, prime);
           // else _InPlaceMultiplyAndHadamardProductWithAcrivationPrime?.Invoke(m, di, wt, prime);
        }

        #endregion
    }
}
