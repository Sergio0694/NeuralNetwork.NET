using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.Network.Activations.Delegates;

namespace NeuralNetworkDotNet.Network.Cost.Delegates
{
    /// <summary>
    /// A <see langword="delegate"/> for a function that computes the derivative of the cost function used to train a neural network
    /// </summary>
    /// <param name="yHat">The current results</param>
    /// <param name="y">The expected results for the dataset</param>
    /// <param name="z">The activity on the last network layer</param>
    /// <param name="activationPrime">The activation pime function for the last network layer</param>
    /// <param name="dx">The backpropagated error</param>
    public delegate void CostFunctionPrime([NotNull] Tensor yHat, [NotNull] Tensor y, [NotNull] Tensor z, [NotNull] ActivationFunction activationPrime, [NotNull] Tensor dx);
}