using JetBrains.Annotations;

namespace NeuralNetworkNET.Networks.Cost
{
    /// <summary>
    /// A delegate that represents a cost function used to compute the accuracy of a neural network
    /// </summary>
    /// <param name="yHat">The output of the network being trained</param>
    /// <param name="y">The expected output for the network</param>
    public delegate float CostFunction([NotNull] float[,] yHat, [NotNull] float[,] y);
}
