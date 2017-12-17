using NeuralNetworkNET.APIs.Interfaces;
using System;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.Adadelta"/> algorithm
    /// </summary>
    public sealed class AdadeltaInfo : ITrainingAlgorithmInfo
    {
        /// <inheritdoc/>
        public TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.Adadelta;

        /// <summary>
        /// Gets the Adadelta rho parameter
        /// </summary>
        public float Rho { get; }

        /// <summary>
        /// Gets the Adadelta epsilon parameter
        /// </summary>
        public float Epsilon { get; }

        /// <summary>
        /// Gets the L2 regularization parameter
        /// </summary>
        public float L2 { get; }

        internal AdadeltaInfo(float rho, float epsilon, float l2)
        {
            Rho = rho;
            Epsilon = epsilon;
            L2 = l2 >= 0 && l2 < 1 ? l2 : throw new ArgumentOutOfRangeException(nameof(l2), "The L2 regularization parameter must be in the [0,1) range");
        }
    }
}
