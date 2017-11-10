using System;
using MnistDatasetToolkit;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace DigitsTest
{
    class Program
    {
        static void Main()
        {
            (var training, var test) = DataParser.LoadDatasets();
            NeuralNetwork network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(100, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            network.StochasticGradientDescent((training.X, training.Y), 10, 10,
                null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
                })),
                0.5f, 5f);
            Console.ReadKey();
        }

        // Prints an output message
        private static void Printf(String text)
        {
            Console.ForegroundColor = ConsoleColor.DarkRed;
            Console.Write(">> ");
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"{text}\n");
        }
    }
}
