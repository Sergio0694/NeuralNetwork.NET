using System;

namespace NeuralNetworkNET.Networks.Architecture
{
    /// <summary>
    /// A static class that holds a reference to the activation functions currently in use
    /// </summary>
    internal static class ActivationFunctionProvider
    {
        /// <summary>
        /// Gets the activation function to use
        /// </summary>
        public static Func<double, double> Activation { get; set; } = ActivationFunctions.Sigmoid;

        /// <summary>
        /// Gets the derivative of the current activation function
        /// </summary>
        public static Func<double, double> ActivationPrime { get; set; } = ActivationFunctions.SigmoidPrime;
    }
}
