using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
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
        public static (ActivationFunction, ActivationFunction) GetActivations(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid: return (ActivationFunctions.Sigmoid, ActivationFunctions.SigmoidPrime);
                case ActivationType.Tanh: return (ActivationFunctions.Tanh, ActivationFunctions.TanhPrime);
                case ActivationType.LeCunTanh: return (ActivationFunctions.LeCunTanh, ActivationFunctions.LeCunTanhPrime);
                case ActivationType.ReLU: return (ActivationFunctions.ReLU, ActivationFunctions.ReLUPrime);
                case ActivationType.LeakyReLU: return (ActivationFunctions.LeakyReLU, ActivationFunctions.LeakyReLUPrime);
                case ActivationType.AbsoluteReLU: return (ActivationFunctions.AbsoluteReLU, ActivationFunctions.AbsoluteReLUPrime);
                case ActivationType.Softmax: return (ActivationFunctions.Softmax, null);
                case ActivationType.Softplus: return (ActivationFunctions.Softplus, ActivationFunctions.Sigmoid);
                case ActivationType.ELU: return (ActivationFunctions.ELU, ActivationFunctions.ELUPrime);
                case ActivationType.Identity: return (ActivationFunctions.Identity, ActivationFunctions.Identityprime);
                default:
                    throw new ArgumentOutOfRangeException(nameof(ActivationType), "Unsupported activation function");
            }
        }
    }
}
