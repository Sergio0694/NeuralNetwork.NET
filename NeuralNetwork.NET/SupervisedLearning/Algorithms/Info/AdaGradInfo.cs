using NeuralNetworkNET.APIs.Interfaces;

namespace NeuralNetworkNET.SupervisedLearning.Algorithms.Info
{
    /// <summary>
    /// A class containing all the info needed to use the <see cref="TrainingAlgorithmType.AdaGrad"/> algorithm
    /// </summary>
    public class AdaGradInfo : ITrainingAlgorithmInfo
    {
        /// <inheritdoc/>
        public virtual TrainingAlgorithmType AlgorithmType => TrainingAlgorithmType.AdaGrad;

        /// <summary>
        /// Gets the current learning rate
        /// </summary>
        public float Eta { get; }

        /// <summary>
        /// Gets the lambda regularization parameter
        /// </summary>
        public float Lambda { get; }

        /// <summary>
        /// Gets the AdaGrad epsilon parameter
        /// </summary>
        public float Epsilon { get; }

        internal AdaGradInfo(float eta, float lambda, float epsilon)
        {
            Eta = eta;
            Lambda = lambda;
            Epsilon = epsilon;
        }
    }
}
