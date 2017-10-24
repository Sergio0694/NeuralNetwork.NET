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
        #region Functional

        /// <summary>
        /// Gets or sets a <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices
        /// </summary>
        [CanBeNull]
        public static Func<double[,], double[,], double[,]> MultiplyOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Multiply"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] Multiply([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return MultiplyOverride?.Invoke(m1, m2) ?? m1.Multiply(m2);
        }

        /// <summary>
        /// Gets or sets a <see cref="Func{T1, T2, TResult}"/> that transposes the first matrix and then multiplies it with the second one
        /// </summary>
        [CanBeNull]
        public static Func<double[,], double[,], double[,]> TransposeAndMultiplyOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Transpose"/> and <see cref="MatrixExtensions.Multiply"/> methods in sequence
        /// </summary>
        [Pure, NotNull]
        public static double[,] TransposeAndMultiply([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return TransposeAndMultiplyOverride?.Invoke(m1, m2) ?? m1.Transpose().Multiply(m2);
        }

        /// <summary>
        /// Gets or sets a <see cref="Func{T, TResult}"/> that applies the sigmoid function
        /// </summary>
        [CanBeNull]
        public static Func<double[,], double[,]> SigmoidOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.Sigmoid"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] Sigmoid([NotNull] double[,] m) => SigmoidOverride?.Invoke(m) ?? m.Sigmoid();

        /// <summary>
        /// Gets or sets a <see cref="Func{T1, T2, TResult}"/> that multiplies two matrices and then applies the sigmoid function
        /// </summary>
        [CanBeNull]
        public static Func<double[,], double[,], double[,]> MultiplyAndSigmoidOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.MultiplyAndSigmoid"/> method
        /// </summary>
        [Pure, NotNull]
        public static double[,] MultiplyAndSigmoid([NotNull] double[,] m1, [NotNull] double[,] m2)
        {
            return MultiplyAndSigmoidOverride?.Invoke(m1, m2) ?? m1.MultiplyAndSigmoid(m2);
        }

        #endregion

        #region Side effect

        /// <summary>
        /// Gets or sets an <see cref="Action{T1, T2, T3}"/> that performs the Hadamard product to the cost function prime, then applies the sigmoid prime function
        /// </summary>
        [CanBeNull]
        public static Action<double[,], double[,], double[,]> InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceSubtractAndHadamardProductWithSigmoidPrime"/> method
        /// </summary>
        public static void InPlaceSubtractAndHadamardProductWithSigmoidPrime([NotNull] double[,] m, [NotNull] double[,] y, [NotNull] double[,] z)
        {
            if (InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride == null) m.InPlaceSubtractAndHadamardProductWithSigmoidPrime(y, z);
            else InPlaceSubtractAndHadamardProductWithSigmoidPrimeOverride?.Invoke(m, y, z);
        }

        /// <summary>
        /// Gets or sets an <see cref="Action{T1, T2, T3}"/> that performs the sigmoid prime function and then the Hadamard product
        /// </summary>
        [CanBeNull]
        public static Action<double[,], double[,]> InPlaceSigmoidPrimeAndHadamardProductOverride { get; set; }

        /// <summary>
        /// Forwards the base <see cref="MatrixExtensions.InPlaceSigmoidPrimeAndHadamardProduct"/> method
        /// </summary>
        public static void InPlaceSigmoidPrimeAndHadamardProduct([NotNull] double[,] m, [NotNull] double[,] delta)
        {
            if (InPlaceSigmoidPrimeAndHadamardProductOverride == null) m.InPlaceSigmoidPrimeAndHadamardProduct(delta);
            else InPlaceSigmoidPrimeAndHadamardProductOverride?.Invoke(m, delta);
        }

        #endregion
    }
}
