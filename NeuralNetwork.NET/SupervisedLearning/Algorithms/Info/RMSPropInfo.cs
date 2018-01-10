using System;
using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.RMSProp"/> algorithm
    /// </summary>
    public sealed class RMSPropInfo : ITrainingAlgorithmInfo
    {
        /// <inheritdoc/>
        public TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.RMSProp;

        /// <summary>
        /// Gets the current learning rate
        /// </summary>
        public float Eta { get; }

        /// <summary>
        /// Gets the RMSProp rho parameter
        /// </summary>
        public float Rho { get; }

        /// <summary>
        /// Gets the lambda regularization parameter
        /// </summary>
        public float Lambda { get; }

        /// <summary>
        /// Gets the RMSProp epsilon parameter
        /// </summary>
        public float Epsilon { get; }

        internal RMSPropInfo(float eta, float rho, float lambda, float epsilon)
        {
            Eta = eta;
            Rho = rho >= 0 && rho < 1 ? rho : throw new ArgumentOutOfRangeException(nameof(rho), "The rho parameter must be in the [0,1) range");
            Lambda = lambda;
            Epsilon = epsilon;
        }
    }
}
