namespace NeuralNetworkNET.SupervisedLearning.Algorithms
{
    /// <summary>
    /// An <see langword="enum"/> that indicates a supervised learning training algorithm
    /// </summary>
    public enum TrainingAlgorithmType : byte
    {
        /// <summary>
        /// The plain stochastic gradient descent training algorithm
        /// </summary>
        StochasticGradientDescent,

        /// <summary>
        /// The Adadelta adaptive learning method, by Matthew D. Zeiler, see <a href="https://arxiv.org/abs/1212.5701">arxiv.org/abs/1212.5701</a>
        /// </summary>
        Adadelta,

        /// <summary>
        /// The Adam learning method, by Diederik P. Kingma and Jimmy Lei Ba, see <a href="https://arxiv.org/pdf/1412.6980v8.pdf">arxiv.org/pdf/1412.6980v8.pdf</a>
        /// </summary>
        Adam,

        /// <summary>
        /// The AdaMax learning method, by Diederik P. Kingma and Jimmy Lei Ba, see section 7.1 of <a href="https://arxiv.org/pdf/1412.6980v8.pdf">arxiv.org/pdf/1412.6980v8.pdf</a>
        /// </summary>
        AdaMax
    }
}
