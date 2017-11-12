using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;

namespace NeuralNetworkNET.Networks.Cost.Delegates
{
    /// <summary>
    /// A delegates for a function that computes the derivative of the cost function used to train a neural network
    /// </summary>
    /// <param name="yHat">The current results</param>
    /// <param name="y">The expected results for the dataset</param>
    /// <param name="z">The activity on the last network layer</param>
    /// <param name="activationPrime">The activation pime function for the last network layer</param>
    public delegate void CostFunctionPrime([NotNull] float[,] yHat, [NotNull] float[,] y, [NotNull] float[,] z, ActivationFunction activationPrime);
}
