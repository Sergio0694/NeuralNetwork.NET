using System;
using JetBrains.Annotations;
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
            [NotNull] Func<double[,], double[,], double[,]> multiply,
            [NotNull] Func<double[,], double[,], double[,]> transposeMultiply,
            [NotNull] Func<double[,], double[,], double[,]> multiplySigmoid,
            [NotNull] Func<double[,], double[,]> sigmoid,
            [NotNull] Action<double[,], double[,], double[,]> inPlaceSubtractHadamardSigmoidPrime,
            [NotNull] Action<double[,], double[,]> inPlaceSigmoidPrimeHadamard)
        {
            _MultiplyOverride = multiply;
            _TransposeAndMultiplyOverride = transposeMultiply;
            _MultiplyAndSigmoidOverride = multiplySigmoid;
            _SigmoidOverride = sigmoid;
            _InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride = inPlaceSubtractHadamardSigmoidPrime;
            _InPlaceSigmoidPrimeAndHadamardProductOverride = inPlaceSigmoidPrimeHadamard;
        }

        /// <summary>
        /// Resets the previous injections and restores the default behavior of the service provider
        /// </summary>
        public static void ResetInjections()
        {
            _MultiplyOverride = _TransposeAndMultiplyOverride = _MultiplyAndSigmoidOverride = null;
            _SigmoidOverride = null;
            _InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride = null;
            _InPlaceSigmoidPrimeAndHadamardProductOverride = null;
        }

        #endregion

        #region Functional

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double[,]> _MultiplyOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Multiply"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] Multiply([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return _MultiplyOverride?.Invoke(m1, m2) ?? m1.Multiply(m2);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that transposes the first matrix and then multiplies it with the second one
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double[,]> _TransposeAndMultiplyOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Transpose"/> and <see cref="MatrixExtensions.Multiply"/> methods in sequence
        /// </summary>
        [Pure, NotNull]
        public static double[,] TransposeAndMultiply([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return _TransposeAndMultiplyOverride?.Invoke(m1, m2) ?? m1.Transpose().Multiply(m2);
        }

        /// <summary>
        /// A <see cref="Func{T, TResult}"/> that applies the sigmoid function
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,]> _SigmoidOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Sigmoid"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] Sigmoid([NotNull] double[,] m) => _SigmoidOverride?.Invoke(m) ?? m.Sigmoid();

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices and then applies the sigmoid function
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double[,]> _MultiplyAndSigmoidOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyAndSigmoid"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] MultiplyAndSigmoid([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return _MultiplyAndSigmoidOverride?.Invoke(m1, m2) ?? m1.MultiplyAndSigmoid(m2);
        }

        #endregion

        #region Side effect

        /// <summary>
        /// An <see cref="Action{T1, T2, T3}"/> that performs the Hadamard product to the cost function prime, then applies the sigmoid prime function
        /// </summary>
        [CanBeNull]
        private static Action<double[,], double[,], double[,]> _InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceSubtractAndHadamardProductWithSigmoidPrime"/> method
        /// </summary>
        public static void InPlaceSubtractAndHadamardProductWithSigmoidPrime([NotNull] double[,] m, [NotNull] double[,] y, [NotNull] double[,] z)
        {
            if (_InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride == null) m.InPlaceSubtractAndHadamardProductWithSigmoidPrime(y, z);
            else _InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride?.Invoke(m, y, z);
        }

        /// <summary>
        /// An <see cref="Action{T1, T2, T3}"/> that performs the sigmoid prime function and then the Hadamard product
        /// </summary>
        [CanBeNull]
        private static Action<double[,], double[,]> _InPlaceSigmoidPrimeAndHadamardProductOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceSigmoidPrimeAndHadamardProduct"/> method
        /// </summary>
        public static void InPlaceSigmoidPrimeAndHadamardProduct([NotNull] double[,] m, [NotNull] double[,] delta)
        {
            if (_InPlaceSigmoidPrimeAndHadamardProductOverride == null) m.InPlaceSigmoidPrimeAndHadamardProduct(delta);
            else _InPlaceSigmoidPrimeAndHadamardProductOverride?.Invoke(m, delta);
        }

        #endregion
    }
}
