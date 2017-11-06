using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;

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
            [NotNull] Func<double[,], double[,], double[], double[,]> multiplyWithSum,
            [NotNull] Func<double[,], double[,], double[,]> transposeMultiply,
            [NotNull] Func<double[,], double[,], ActivationFunction, double[,]> multiplyActivation,
            [NotNull] Func<double[,], double[,], double[], ActivationFunction, double[,]> multiplyWithSumAndActivation,
            [NotNull] Func<double[,], ActivationFunction, double[,]> activation,
            [NotNull] Func<double[,], double[,], double> halfSquaredDifference,
            [NotNull] Action<double[,], double[,], double[,], ActivationFunction> inPlaceSubtractHadamardActivationPrime,
            [NotNull] Action<double[,], double[,], double[,], ActivationFunction> multiplyAndInPlaceActivationPrimeHadamard)
        {
            _MultiplyOverride = multiply;
            _MultiplyWithSumOverride = multiplyWithSum;
            _TransposeAndMultiplyOverride = transposeMultiply;
            _MultiplyAndActivationOverride = multiplyActivation;
            _MultiplyWithSumAndActivationOverride = multiplyWithSumAndActivation;
            _ActivationOverride = activation;
            _HalfSquaredDifferenceOverride = halfSquaredDifference;
            _InPlaceSubtractAndHadamardProductWithActivationPrimeOverride = inPlaceSubtractHadamardActivationPrime;
            _MultiplyAndInPlaceActivationPrimeAndHadamardProductOverride = multiplyAndInPlaceActivationPrimeHadamard;
        }

        /// <summary>
        /// Resets the previous injections and restores the default behavior of the service provider
        /// </summary>
        public static void ResetInjections()
        {
            _MultiplyOverride = _TransposeAndMultiplyOverride = null;
            _MultiplyAndActivationOverride = null;
            _ActivationOverride = null;
            _HalfSquaredDifferenceOverride = null;
            _InPlaceSubtractAndHadamardProductWithActivationPrimeOverride = null;
            _MultiplyAndInPlaceActivationPrimeAndHadamardProductOverride = null;
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
        /// A <see cref="Func{T1, T2, T3, TResult}"/> that multiplies two matrices and sums the input vector
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double[], double[,]> _MultiplyWithSumOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSum"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] MultiplyWithSum([NotNull] double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v)
        {
            return _MultiplyWithSumOverride?.Invoke(m1, m2, v) ?? m1.MultiplyWithSum(m2, v);
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
        /// A <see cref="Func{T, TResult}"/> that applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<double[,], ActivationFunction, double[,]> _ActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Activation"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] Activation([NotNull] double[,] m, [NotNull] ActivationFunction activation)
        {
            return _ActivationOverride?.Invoke(m, activation) ?? m.Activation(activation);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices and then applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], ActivationFunction, double[,]> _MultiplyAndActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyAndActivation"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] MultiplyAndActivation([NotNull] double[,] m1, [NotNull] double[,] m2, [NotNull] ActivationFunction activation)
        {
            return _MultiplyAndActivationOverride?.Invoke(m1, m2, activation) ?? m1.MultiplyAndActivation(m2, activation);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, T3, TResult}"/> that multiplies two matrices, sums the input vector and then applies the activation function
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double[], ActivationFunction, double[,]> _MultiplyWithSumAndActivationOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyWithSumAndActivation"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] MultiplyWithSumAndActivation([NotNull] double[,] m1, [NotNull] double[,] m2, [NotNull] double[] v, [NotNull] ActivationFunction activation)
        {
            return _MultiplyWithSumAndActivationOverride?.Invoke(m1, m2, v, activation) ?? m1.MultiplyWithSumAndActivation(m2, v, activation);
        }

        /// <summary>
        /// A <see cref="Func{T1, T2, TResult}"/> that calculates half the squared difference of two matrices
        /// </summary>
        [CanBeNull]
        private static Func<double[,], double[,], double> _HalfSquaredDifferenceOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.HalfSquaredDifference"/> method
        /// </summary>
        [Pure]
        public static double HalfSquaredDifference([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return _HalfSquaredDifferenceOverride?.Invoke(m1, m2) ?? m1.HalfSquaredDifference(m2);
        }

        #endregion

        #region Side effect

        /// <summary>
        /// An <see cref="Action{T1, T2, T3}"/> that performs the Hadamard product to the cost function prime, then applies the activation prime function
        /// </summary>
        [CanBeNull]
        private static Action<double[,], double[,], double[,], ActivationFunction> _InPlaceSubtractAndHadamardProductWithActivationPrimeOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceSubtractAndHadamardProductWithActivationPrime"/> method
        /// </summary>
        public static void InPlaceSubtractAndHadamardProductWithActivationPrime([NotNull] double[,] m, [NotNull] double[,] y, [NotNull] double[,] z, [NotNull] ActivationFunction prime)
        {
            if (_InPlaceSubtractAndHadamardProductWithActivationPrimeOverride == null) m.InPlaceSubtractAndHadamardProductWithActivationPrime(y, z, prime);
            else _InPlaceSubtractAndHadamardProductWithActivationPrimeOverride?.Invoke(m, y, z, prime);
        }

        /// <summary>
        /// An <see cref="Action{T1, T2, T3}"/> that performs the activation prime function and then the Hadamard product with a matrix product
        /// </summary>
        [CanBeNull]
        private static Action<double[,], double[,], double[,], ActivationFunction> _MultiplyAndInPlaceActivationPrimeAndHadamardProductOverride;

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyAndInPlaceActivationPrimeAndHadamardProduct"/> method
        /// </summary>
        public static void MultiplyAndInPlaceActivationPrimeAndHadamardProduct([NotNull] double[,] m, [NotNull] double[,] di, [NotNull] double[,] wt, [NotNull] ActivationFunction prime)
        {
            if (_MultiplyAndInPlaceActivationPrimeAndHadamardProductOverride == null) m.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(di, wt, prime);
            else _MultiplyAndInPlaceActivationPrimeAndHadamardProductOverride?.Invoke(m, di, wt, prime);
        }

        #endregion
    }
}
