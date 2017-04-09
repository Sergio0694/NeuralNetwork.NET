using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks;

namespace NeuralNetworkNET.SupervisedLearning
{
    /// <summary>
    /// A structure that contains the base progress data while optimizing a network
    /// </summary>
    public sealed class BackpropagationProgressEventArgs<T> : EventArgs where T : NeuralNetworkBase
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
        private readonly Func<T> NetworkFactory;

        [CanBeNull]
        private T _Network;

        /// <summary>
        /// Gets the current network for the optimization iteration (lazy evaluation)
        /// </summary>
        [NotNull]
        public T Network => _Network ?? (_Network = NetworkFactory());

        /// <summary>
        /// Internal constructor for the event args base
        /// </summary>
        /// <param name="networkFactory">The factory that will produce a lazy-evaluated neural network for the current iteration</param>
        /// <param name="iteration">The current iteration</param>
        /// <param name="cost">The current function cost</param>
        internal BackpropagationProgressEventArgs([NotNull] Func<T> networkFactory, int iteration, double cost)
        {
            NetworkFactory = networkFactory;
            Iteration = iteration;
            Cost = cost;
        }
    }
}
