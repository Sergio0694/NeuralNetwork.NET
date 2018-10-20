using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Algorithms;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// A common interface for all the available training algorithms in the library
    /// </summary>
    [PublicAPI]
    public interface ITrainingAlgorithmInfo
    {
        /// <summary>
        /// Gets the type of training algorithm for the current instance
        /// </summary>
        TrainingAlgorithmType AlgorithmType { get; }
    }
}
