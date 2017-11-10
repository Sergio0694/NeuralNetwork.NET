using JetBrains.Annotations;

namespace NeuralNetworkNET.UnsupervisedLearning
{
    /// <summary>
    /// Represents the forward method used to process the input data using a neural network
    /// </summary>
    /// <param name="input">The input data to process</param>
    [NotNull]
    public delegate float[,] ForwardFunction([NotNull] float[,] input);
}
