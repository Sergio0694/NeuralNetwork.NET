using JetBrains.Annotations;
using NeuralNetworkNET.SupervisedLearning.Algorithms.Info;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that produces info for different available training algorithms
    /// </summary>
    public static class TrainingAlgorithms
    {
        /// <summary>
        /// Gets an instance implementing <see cref="Interfaces.ITrainingAlgorithmInfo"/> for the <see cref="SupervisedLearning.Algorithms.TrainingAlgorithmType.StochasticGradientDescent"/> algorithm
        /// </summary>
        /// <param name="eta">The learning rate</param>
        /// <param name="lambda">The lambda regularization parameter</param>
        [PublicAPI]
        [Pure, NotNull]
        public static StochasticGradientDescentInfo StochasticGradientDescent(float eta = 0.1f, float lambda = 0f) => new StochasticGradientDescentInfo(eta, lambda);

        /// <summary>
        /// Gets an instance implementing <see cref="Interfaces.ITrainingAlgorithmInfo"/> for the <see cref="SupervisedLearning.Algorithms.TrainingAlgorithmType.Adadelta"/> algorithm
        /// </summary>
        /// <param name="rho">The Adadelta rho parameter</param>
        /// <param name="epsilon">The Adadelta epsilon parameter</param>
        /// <param name="l2">An optional L2 regularization parameter</param>
        [PublicAPI]
        [Pure, NotNull]
        public static AdadeltaInfo Adadelta(float rho = 0.95f, float epsilon = 1e-8f, float l2 = 0f) => new AdadeltaInfo(rho, epsilon, l2);
    }
}
