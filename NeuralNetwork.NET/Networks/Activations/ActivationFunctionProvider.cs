using System;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Networks.Activations
{
    /// <summary>
    /// A static class that returns the right activation function for the given type
    /// </summary>
    internal static class ActivationFunctionProvider
    {
        /// <summary>
        /// Gets an activation function for the given type
        /// </summary>
        /// <param name="type">The activation function type</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ActivationFunction GetActivation(ActivationFunctionType type)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid: return ActivationFunctions.Sigmoid;
                case ActivationFunctionType.Tanh: return ActivationFunctions.Tanh;
                case ActivationFunctionType.ReLU: return ActivationFunctions.ReLU;
                case ActivationFunctionType.LeakyReLU: return ActivationFunctions.LeakyReLU;
                case ActivationFunctionType.Softplus: return ActivationFunctions.Softplus;
                case ActivationFunctionType.ELU: return ActivationFunctions.ELU;
                default:
                    throw new ArgumentOutOfRangeException(nameof(ActivationFunctionType), "Unsupported activation function");
            }
        }

        /// <summary>
        /// Gets the derivative of the activation function requested
        /// </summary>
        /// <param name="type">The activation function type</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ActivationFunction GetActivationPrime(ActivationFunctionType type)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid: return ActivationFunctions.SigmoidPrime;
                case ActivationFunctionType.Tanh: return ActivationFunctions.TanhPrime;
                case ActivationFunctionType.ReLU: return ActivationFunctions.ReLUPrime;
                case ActivationFunctionType.LeakyReLU: return ActivationFunctions.LeakyReLUPrime;
                case ActivationFunctionType.Softplus: return ActivationFunctions.Sigmoid;
                case ActivationFunctionType.ELU: return ActivationFunctions.ELUPrime;
                default:
                    throw new ArgumentOutOfRangeException(nameof(ActivationFunctionType), "Unsupported activation function");
            }
        }
    }
}
