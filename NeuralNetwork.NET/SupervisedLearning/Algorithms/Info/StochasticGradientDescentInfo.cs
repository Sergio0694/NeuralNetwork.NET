using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.StochasticGradientDescent"/> algorithm
    /// </summary>
    public class StochasticGradientDescentInfo : ITrainingAlgorithmInfo
    {
        /// <inheritdoc/>
        public virtual TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.StochasticGradientDescent;

        /// <summary>
        /// Gets the current learning rate
        /// </summary>
        public float Eta { get; }

        /// <summary>
        /// Gets the lambda regularization parameter
        /// </summary>
        public float Lambda { get; }

        internal StochasticGradientDescentInfo(float eta, float lambda)
        {
            Eta = eta;
            Lambda = lambda;
        }
    }
}
