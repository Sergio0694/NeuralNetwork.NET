namespace NeuralNetworkNET.SupervisedLearning.Algorithms
{
    /// <summary>
    /// An <see cref="enum"/> that indicates a supervised learning training algorithm
    /// </summary>
    public enum TrainingAlgorithmType : byte
    {
        /// <summary>
        /// The plain stochastic gradient descent training algorithm
        /// </summary>
        StochasticGradientDescent,

        /// <summary>
        /// The Adadelta adaptive learning method, by Matthew D. Zeiler, see <see cref="https://arxiv.org/abs/1212.5701"/>
        /// </summary>
        Adadelta
    }
}
