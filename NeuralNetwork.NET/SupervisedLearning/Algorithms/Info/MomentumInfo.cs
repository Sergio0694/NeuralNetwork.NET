using System;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.Momentum"/> algorithm
    /// </summary>
    public sealed class MomentumInfo : StochasticGradientDescentInfo
    {
        /// <inheritdoc/>
        public override TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.Momentum;

        /// <summary>
        /// Gets the current momentum
        /// </summary>
        public float Momentum { get; }

        internal MomentumInfo(float eta, float lambda, float momentum) : base(eta, lambda) 
            => Momentum = momentum >= 0 && momentum < 1 ? momentum : throw new ArgumentOutOfRangeException(nameof(momentum), "The momentum parameter must be in the [0,1) range");
    }
}
