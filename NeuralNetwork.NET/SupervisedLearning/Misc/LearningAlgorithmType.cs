namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// Indicates the type of learning algorithm to use to train a neural network
    /// </summary>
    public enum LearningAlgorithmType
    {
        /// <summary>
        /// Gradient descend optimization algorithm
        /// </summary>
        GradientDescend,

        /// <summary>
        /// Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizaation algorithm
        /// </summary>
        BoundedFGS,

        /// <summary>
        /// Limited-memory Broyden–Fletcher–Goldfarb–Shanno and then gradient descent after first convergence
        /// </summary>
        BoundedBFGSWithGradientDescentOnFirstConvergence
    }
}