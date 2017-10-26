using System;
using NeuralNetworkNET.Networks.Architecture;

namespace NeuralNetworkNET.Networks.PublicAPIs
{
    /// <summary>
    /// A static class with some additional settings for the neural networks produced by the library
    /// </summary>
    public static class NeuralNetworkSettings
    {
        private static ActivationFunction _ActivationFunctionType;

        /// <summary>
        /// Gets or sets the activation function to use in the neural networks
        /// </summary>
        public static ActivationFunction ActivationFunctionType
        {
            get => _ActivationFunctionType;
            set
            {
                if (_ActivationFunctionType != value)
                {
                    switch (value)
                    {
                        case ActivationFunction.Sigmoid:
                            ActivationFunctionProvider.Activation = ActivationFunctions.Sigmoid;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.SigmoidPrime;
                            break;
                        case ActivationFunction.Tanh:
                            ActivationFunctionProvider.Activation = ActivationFunctions.Tanh;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.TanhPrime;
                            break;
                        case ActivationFunction.ReLU:
                            ActivationFunctionProvider.Activation = ActivationFunctions.ReLU;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.ReLUPrime;
                            break;
                        case ActivationFunction.LeakyReLU:
                            ActivationFunctionProvider.Activation = ActivationFunctions.LeakyReLU;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.LeakyReLUPrime;
                            break;
                        case ActivationFunction.Softplus:
                            ActivationFunctionProvider.Activation = ActivationFunctions.Softplus;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.Sigmoid;
                            break;
                        case ActivationFunction.ELU:
                            ActivationFunctionProvider.Activation = ActivationFunctions.ELU;
                            ActivationFunctionProvider.ActivationPrime = ActivationFunctions.ELUPrime;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(ActivationFunction), "Unsupported activation function");
                    }
                    _ActivationFunctionType = value;
                }
            }
        }
    }
}
