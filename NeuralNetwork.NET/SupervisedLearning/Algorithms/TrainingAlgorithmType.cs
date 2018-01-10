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
        /// A variant of the stochastic gradient descent algorithm with momentum
        /// </summary>
        Momentum,

        /// <summary>
        /// The AdaGrad learning method, by John Duchi, Elad Hazan and Yoram Singer, see <a href="http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf">jmlr.org/papers/volume12/duchi11a/duchi11a.pdf</a>
        /// </summary>
        AdaGrad,

        /// <summary>
        /// The AdaDelta adaptive learning method, by Matthew D. Zeiler, see <a href="https://arxiv.org/abs/1212.5701">arxiv.org/abs/1212.5701</a>
        /// </summary>
        AdaDelta,

        /// <summary>
        /// The Adam learning method, by Diederik P. Kingma and Jimmy Lei Ba, see <a href="https://arxiv.org/pdf/1412.6980v8.pdf">arxiv.org/pdf/1412.6980v8.pdf</a>
        /// </summary>
        Adam,

        /// <summary>
        /// The AdaMax learning method, by Diederik P. Kingma and Jimmy Lei Ba, see section 7.1 of <a href="https://arxiv.org/pdf/1412.6980v8.pdf">arxiv.org/pdf/1412.6980v8.pdf</a>
        /// </summary>
        AdaMax,

        /// <summary>
        /// The RMSProp learning method, see <a href="http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf">cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf</a>
        /// </summary>
        RMSProp
    }
}
