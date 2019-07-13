using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;

namespace NeuralNetworkDotNet.Network.Cost.Delegates
{
    /// <summary>
    /// A <see langword="delegate"/> that represents a cost function used to compute the accuracy of a neural network
    /// </summary>
    /// <param name="yHat">The output of the network being trained</param>
    /// <param name="y">The expected output for the network</param>
    public delegate float CostFunction([NotNull] Tensor yHat, [NotNull] Tensor y);
}
