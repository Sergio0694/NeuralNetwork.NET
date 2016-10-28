using System;
using NeuralNetworkLibrary.Networks.PublicAPIs;

namespace NeuralNetworkLibrary.GeneticAlgorithm.Misc
{
    /// <summary>
    /// The argument for the BestSpeciesChanged event
    /// </summary>
    public class GeneticAlgorithmBestNetworkChangedEventArgs : EventArgs
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
