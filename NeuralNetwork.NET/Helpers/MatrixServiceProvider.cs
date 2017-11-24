using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;

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
            [NotNull] Func<float[,], int, float[,], int, float[,]> convoluteForward,
            [NotNull] Func<float[,], int, float[,], int, float[,]> convoluteBackwards,
            [NotNull] Func<float[,], int, float[,], int, float[,]> convoluteGradient)
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
            _ConvoluteForwardOverride = _ConvoluteBackwardsOverride = _ConvoluteGradientOverride = null;
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
        public static float[,] MultiplyWithSum([NotNull] float[,] m1, [NotNull] float[,] m2, [NotNull] float[] v)
        {
            return _MultiplyWithSumOverride?.Invoke(m1, m2, v) ?? m1.MultiplyWithSum(m2, v);
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
        public static float[,] TransposeAndMultiply([NotNull] float[,] m1, [NotNull] float[,] m2)
        {
            return _TransposeAndMultiplyOverride?.Invoke(m1, m2) ?? m1.Transpose().Multiply(m2);
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
        public static float[,] Activation([NotNull] float[,] m, [NotNull] ActivationFunction activation)
        {
            return _ActivationOverride?.Invoke(m, activation) ?? m.Activation(activation);
        }

        #endregion

        #region Convolution

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, T4, TResult}"/> that performs the forward convolution
        /// </summary>
        [CanBeNull]
        private static Func<float[,], int, float[,], int, float[,]> _ConvoluteForwardOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteForward"/> method
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,] ConvoluteForward([NotNull] float[,] m1, VolumeInformation m1Info, [NotNull] float[,] m2, VolumeInformation m2Info)
        {
            return m1.ConvoluteForward(m1Info, m2, m2Info);
            //return _ConvoluteForwardOverride?.Invoke(m1, m1depth, m2, m2depth) ?? m1.ConvoluteForward(m1depth, m2, m2depth);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, T4, TResult}"/> that performs the backwards convolution
        /// </summary>
        [CanBeNull]
        private static Func<float[,], int, float[,], int, float[,]> _ConvoluteBackwardsOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteBackwards"/> method
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,] ConvoluteBackwards([NotNull] float[,] m1, VolumeInformation m1Info, [NotNull] float[,] m2, VolumeInformation m2Info)
        {
            return m1.ConvoluteBackwards(m1Info, m2, m2Info);
            //return _ConvoluteBackwardsOverride?.Invoke(m1, m1depth, m2, m2depth) ?? m1.ConvoluteBackwards(m1depth, m2, m2depth);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, T4, TResult}"/> that performs the gradient convolution
        /// </summary>
        [CanBeNull]
        private static Func<float[,], int, float[,], int, float[,]> _ConvoluteGradientOverride;

        /// <summary>
        /// Forwards the base <see cref="ConvolutionExtensions.ConvoluteGradient"/> method
        /// </summary>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[,] ConvoluteGradient([NotNull] float[,] m1, VolumeInformation m1Info, [NotNull] float[,] m2, VolumeInformation m2Info)
        {
            return m1.ConvoluteGradient(m1Info, m2, m2Info);
            //return _ConvoluteGradientOverride?.Invoke(m1, m1depth, m2, m2depth) ?? m1.ConvoluteGradient(m1depth, m2, m2depth);
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
        public static void InPlaceMultiplyAndHadamardProductWithActivationPrime([NotNull] float[,] m, [NotNull] float[,] di, [NotNull] float[,] wt, [NotNull] ActivationFunction prime)
        {
            if (_InPlaceMultiplyAndHadamardProductWithAcrivationPrime == null) m.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(di, wt, prime);
            else _InPlaceMultiplyAndHadamardProductWithAcrivationPrime?.Invoke(m, di, wt, prime);
        }

        #endregion
    }
}
