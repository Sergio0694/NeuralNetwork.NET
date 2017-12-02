using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.DependencyInjection.Delegates;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.DependencyInjection
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
            [NotNull] MultiplicationWithSum multiplyWithSum,
            [NotNull] Multiplication transposeMultiply,
            [NotNull] Activation activation,
            [NotNull] MultiplicationAndHadamardProductWithActivation multiplyAndInPlaceActivationPrimeHadamard,
            [NotNull] ConvolutionWithBias convoluteForward,
            [NotNull] Convolution convoluteBackwards,
            [NotNull] Convolution convoluteGradient)
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
            _ConvoluteBackwardsOverride = _ConvoluteGradientOverride = null;
        }

        #endregion

        #region Functional
        
        [CanBeNull]
        private static MultiplicationWithSum _MultiplyWithSumOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSum"/> method
        /// </summary>
        [MustUseReturnValue]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyWithSum(in FloatSpan2D m1, float[,] m2, float[] v, out FloatSpan2D result)
        {
            if (_MultiplyWithSumOverride == null) m1.MultiplyWithSum(m2, v, out result);
            else _MultiplyWithSumOverride(m1, m2, v, out result);
        }

        [CanBeNull]
        private static Multiplication _TransposeAndMultiplyOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Transpose"/> and <see cref="MatrixExtensions.Multiply"/> methods in sequence
        /// </summary>
        [MustUseReturnValue]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TransposeAndMultiply(in FloatSpan2D m1, in FloatSpan2D m2, out FloatSpan2D result)
        {
            if (_TransposeAndMultiplyOverride == null)
            {
                m1.Transpose(out FloatSpan2D m1t);
                m1t.Multiply(m2, out result);
                m1t.Free();
            }
            else _TransposeAndMultiplyOverride(m1, m2, out result);
        }
        
        [CanBeNull]
        private static Activation _ActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Activation"/> method
        /// </summary>
        [MustUseReturnValue]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Activation(in FloatSpan2D m, [NotNull] ActivationFunction activation, out FloatSpan2D result)
        {
            if (_ActivationOverride == null) m.Activation(activation, out result);
            else _ActivationOverride(m, activation, out result);
        }

        #endregion

        #region Convolution

        /// <summary>
        /// A <see cref="ForwardConvolution"/> delegate that performs the forward convolution
        /// </summary>
        [CanBeNull]
        private static ConvolutionWithBias _ConvoluteForwardOverride;

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
        /// A <see cref="Convolution"/> dekegate that performs the backwards convolution
        /// </summary>
        [CanBeNull]
        private static Convolution _ConvoluteBackwardsOverride;

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
        /// A <see cref="Convolution"/> function that performs the gradient convolution
        /// </summary>
        [CanBeNull]
        private static Convolution _ConvoluteGradientOverride;

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
        
        [CanBeNull]
        private static MultiplicationAndHadamardProductWithActivation _InPlaceMultiplyAndHadamardProductWithAcrivationPrime;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceMultiplyAndHadamardProductWithActivationPrime"/> method
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void InPlaceMultiplyAndHadamardProductWithActivationPrime(in FloatSpan2D m, in FloatSpan2D di, in FloatSpan2D wt, [NotNull] ActivationFunction prime)
        {
            if (_InPlaceMultiplyAndHadamardProductWithAcrivationPrime == null) m.InPlaceMultiplyAndHadamardProductWithActivationPrime(di, wt, prime);
            else _InPlaceMultiplyAndHadamardProductWithAcrivationPrime(m, di, wt, prime);
        }

        #endregion
    }
}
