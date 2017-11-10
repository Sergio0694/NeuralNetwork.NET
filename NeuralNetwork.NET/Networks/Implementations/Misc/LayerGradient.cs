using JetBrains.Annotations;

namespace NeuralNetworkNET.Networks.Implementations.Misc
{
    /// <summary>
    /// A simple struct that holds the gradient for each layer in a neural network
    /// </summary>
    internal struct LayerGradient
    {
        /// <summary>
        /// Gets the gradient with respect to the current weights
        /// </summary>
        public double[,] DJdw { get; }

        /// <summary>
        /// Gets the gradient with respect to the current biases
        /// </summary>
        public double[] Djdb { get; }

        public LayerGradient([NotNull] double[,] dJdw, [NotNull] double[] dJdb)
        {
            DJdw = dJdw;
            Djdb = dJdb;
        }
    }
}