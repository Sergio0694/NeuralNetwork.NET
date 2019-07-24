using System;
using System.Threading;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.ReinforcedLearning.Environments;

namespace _2048Test
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create the network
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Linear(16),
                NetworkLayers.FullyConnected(16, ActivationType.LeakyReLU),
                NetworkLayers.FullyConnected(16, ActivationType.LeakyReLU),
                NetworkLayers.FullyConnected(4, ActivationType.ReLU, CostFunctionType.Quadratic));

            // Create the environment
            _2048Environment environment = new _2048Environment();

            // Train the network
            NetworkManager.TrainNetwork(
                network,
                environment,
                100, 0.9f,
                score => Console.WriteLine($"SCORE: {score}"),
                CancellationToken.None);
        }
    }
}
