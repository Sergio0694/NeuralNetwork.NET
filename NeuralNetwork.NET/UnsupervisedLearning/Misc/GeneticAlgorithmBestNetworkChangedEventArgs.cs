using System;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.UnsupervisedLearning.Misc
{
    /// <summary>
    /// The argument for the BestSpeciesChanged event
    /// </summary>
    public sealed class GeneticAlgorithmBestNetworkChangedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets the generated neural network
        /// </summary>
        public INeuralNetwork Network { get; }

        /// <summary>
        /// Gets the fitness score obtained by the neural network
        /// </summary>
        public double Fitness { get; }

        // Internal constructor
        internal GeneticAlgorithmBestNetworkChangedEventArgs(INeuralNetwork network, double fitess)
        {
            Network = network;
            Fitness = fitess;
        }
    }
}
