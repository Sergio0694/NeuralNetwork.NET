namespace NeuralNetworkNET.Networks.Cost
{
    /// <summary>
    /// Indicates the cost function to use to evaluate and train a neural network
    /// </summary>
    public enum CostFunctionType : byte
    {
        /// <summary>
        /// The classic quadratic cost function, 1/n(yHat - y)^2
        /// </summary>
        Quadratic,

        /// <summary>
        /// The cross-entropy cost function, -1/n(y*ln(yHat) + (1 - y)ln(1 - yHat))
        /// </summary>
        CrossEntropy,

        /// <summary>
        /// The log-likelyhood cost function (for a softmax output layer), -ln(yHat[y])
        /// </summary>
        LogLikelyhood
    }
}