using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.Networks.Activations
{
    /// <summary>
    /// A static class that returns the right activation function for the given type
    /// </summary>
    internal static class ActivationFunctionProvider
    {
        /// <summary>
        /// Gets an activation and activation prime functions for the given type
        /// </summary>
        /// <param name="type">The activation function type</param>
        [Pure]
        public static (ActivationFunction, ActivationFunction) GetActivations(ActivationFunctionType type)
        {
            switch (type)
            {
                case ActivationFunctionType.Sigmoid: return (ActivationFunctions.Sigmoid, ActivationFunctions.SigmoidPrime);
                case ActivationFunctionType.Tanh: return (ActivationFunctions.Tanh, ActivationFunctions.TanhPrime);
                case ActivationFunctionType.LeCunTanh: return (ActivationFunctions.LeCunTanh, ActivationFunctions.LeCunTanhPrime);
                case ActivationFunctionType.ReLU: return (ActivationFunctions.ReLU, ActivationFunctions.ReLUPrime);
                case ActivationFunctionType.LeakyReLU: return (ActivationFunctions.LeakyReLU, ActivationFunctions.LeakyReLUPrime);
                case ActivationFunctionType.AbsoluteReLU: return (ActivationFunctions.AbsoluteReLU, ActivationFunctions.AbsoluteReLUPrime);
                case ActivationFunctionType.Softmax: return (ActivationFunctions.Softmax, null);
                case ActivationFunctionType.Softplus: return (ActivationFunctions.Softplus, ActivationFunctions.Sigmoid);
                case ActivationFunctionType.ELU: return (ActivationFunctions.ELU, ActivationFunctions.ELUPrime);
                case ActivationFunctionType.Identity: return (ActivationFunctions.Identity, ActivationFunctions.Identityprime);
                default:
                    throw new ArgumentOutOfRangeException(nameof(ActivationFunctionType), "Unsupported activation function");
            }
        }
    }
}
