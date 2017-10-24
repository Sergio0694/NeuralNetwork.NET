using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A structure that contains the base progress data while optimizing a network
    /// </summary>
    public sealed class BackpropagationProgressEventArgs
    {
        /// <summary>
        /// Gets the current iteration number
        /// </summary>
        public int Iteration { get; }

        /// <summary>
        /// Gets the current value for the function to optimize
        /// </summary>
        public double Cost { get; }

        // Factory for the network lazy evaluation
        [NotNull]
        private readonly Func<INeuralNetwork> NetworkFactory;

        [CanBeNull]
        private INeuralNetwork _Network;

        /// <summary>
        /// Gets the current network for the optimization iteration (lazy evaluation)
        /// </summary>
        [NotNull]
        public INeuralNetwork Network => _Network ?? (_Network = NetworkFactory());

        /// <summary>
        /// Internal constructor for the event args base
        /// </summary>
        /// <param name="networkFactory">The factory that will produce a lazy-evaluated neural network for the current iteration</param>
        /// <param name="iteration">The current iteration</param>
        /// <param name="cost">The current function cost</param>
        internal BackpropagationProgressEventArgs([NotNull] Func<INeuralNetwork> networkFactory, int iteration, double cost)
        {
            NetworkFactory = networkFactory;
            Iteration = iteration;
            Cost = cost;
        }
    }
}
