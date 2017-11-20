using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;

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
            [NotNull] Func<float[,], float[,], float[,]> multiply,
            [NotNull] Func<float[,], float[,], float[], float[,]> multiplyWithSum,
            [NotNull] Func<float[,], float[,], float[,]> transposeMultiply,
            [NotNull] Func<float[,], float[,], ActivationFunction, float[,]> multiplyActivation,
            [NotNull] Func<float[,], float[,], float[], ActivationFunction, float[,]> multiplyWithSumAndActivation,
            [NotNull] Func<float[,], ActivationFunction, float[,]> activation,
            [NotNull] Action<float[,], float[,], float[,], ActivationFunction> multiplyAndInPlaceActivationPrimeHadamard)
        {
            _MultiplyOverride = multiply;
            _MultiplyWithSumOverride = multiplyWithSum;
            _TransposeAndMultiplyOverride = transposeMultiply;
            _MultiplyAndActivationOverride = multiplyActivation;
            _MultiplyWithSumAndActivationOverride = multiplyWithSumAndActivation;
            _ActivationOverride = activation;
            _InPlaceMultiplyAndHadamardProductWithAcrivationPrime = multiplyAndInPlaceActivationPrimeHadamard;
        }

        /// <summary>
        /// Resets the previous injections and restores the default behavior of the service provider
        /// </summary>
        public static void ResetInjections()
        {
            _MultiplyOverride = _TransposeAndMultiplyOverride = null;
            _MultiplyAndActivationOverride = null;
            _ActivationOverride = null;
            _InPlaceMultiplyAndHadamardProductWithAcrivationPrime = null;
        }

        #endregion

        #region Functional

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], float[,]> _MultiplyOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Multiply"/> method
        /// </summary>
        [Pure, NotNull]
        public static float[,] Multiply([NotNull] float[,] m1, [NotNull] float[,] m2)
        {
            return _MultiplyOverride?.Invoke(m1, m2) ?? m1.Multiply(m2);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, TResult}"/> that multiplies two matrices and sums the input vector
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], float[], float[,]> _MultiplyWithSumOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSum"/> method
        /// </summary>
        [Pure, NotNull]
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
        public static float[,] Activation([NotNull] float[,] m, [NotNull] ActivationFunction activation)
        {
            return _ActivationOverride?.Invoke(m, activation) ?? m.Activation(activation);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices and then applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], ActivationFunction, float[,]> _MultiplyAndActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyAndActivation"/> method
        /// </summary>
        [Pure, NotNull]
        public static float[,] MultiplyAndActivation([NotNull] float[,] m1, [NotNull] float[,] m2, [NotNull] ActivationFunction activation)
        {
            return _MultiplyAndActivationOverride?.Invoke(m1, m2, activation) ?? m1.MultiplyAndActivation(m2, activation);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, TResult}"/> that multiplies two matrices, sums the input vector and then applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<float[,], float[,], float[], ActivationFunction, float[,]> _MultiplyWithSumAndActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSumAndActivation"/> method
        /// </summary>
        [Pure, NotNull]
        public static float[,] MultiplyWithSumAndActivation([NotNull] float[,] m1, [NotNull] float[,] m2, [NotNull] float[] v, [NotNull] ActivationFunction activation)
        {
            return _MultiplyWithSumAndActivationOverride?.Invoke(m1, m2, v, activation) ?? m1.MultiplyWithSumAndActivation(m2, v, activation);
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
        public static void InPlaceMultiplyAndHadamardProductWithActivationPrime([NotNull] float[,] m, [NotNull] float[,] di, [NotNull] float[,] wt, [NotNull] ActivationFunction prime)
        {
            if (_InPlaceMultiplyAndHadamardProductWithAcrivationPrime == null) m.InPlaceMultiplyAndHadamardProductWithAcrivationPrime(di, wt, prime);
            else _InPlaceMultiplyAndHadamardProductWithAcrivationPrime?.Invoke(m, di, wt, prime);
        }

        #endregion
    }
}
