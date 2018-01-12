using System;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.Adam"/> algorithm
    /// </summary>
    public sealed class AdamInfo : ITrainingAlgorithmInfo
    {
        /// <inheritdoc/>
        public TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.Adam;

        /// <summary>
        /// Gets the learning rate factor
        /// </summary>
        public float Eta { get; }

        /// <summary>
        /// Gets the beta1 factor
        /// </summary>
        public float Beta1 { get; }

        /// <summary>
        /// Gets the beta2 factor
        /// </summary>
        public float Beta2 { get; }

        /// <summary>
        /// Gets the Adam epsilon parameter
        /// </summary>
        public float Epsilon { get; }

        internal AdamInfo(float eta, float beta1, float beta2, float epsilon)
        {
            Eta = eta;
            Beta1 = beta1 >= 0 && beta1 < 1 ? beta1 : throw new ArgumentOutOfRangeException(nameof(beta1), "The beta1 factor must be in the [0,1) range");
            Beta2 = beta2 >= 0 && beta2 < 1 ? beta2 : throw new ArgumentOutOfRangeException(nameof(beta2), "The beta2 factor must be in the [0,1) range");
            Epsilon = epsilon;
        }
    }
}
