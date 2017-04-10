using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.UnsupervisedLearning
{
    /// <summary>
    /// Represents the method used to calculate the fitness score for each neural network
    /// </summary>
    /// <param name="uid">A unique identifier for the network</param>
    /// <param name="forwardFunction">The forward function to test the current neural network</param>
    /// <param name="opponents">A collection of the forward functions of the other species in the current generation, used for competitive learning</param>
    public delegate double FitnessDelegate(int uid, [NotNull] ForwardFunction forwardFunction, [NotNull, ItemNotNull] IEnumerable<ForwardFunction> opponents);
}
