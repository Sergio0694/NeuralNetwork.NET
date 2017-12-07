using System;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace DigitsTest
{
    class Program
    {
        static async Task Main()
        {
            (var training, var test) = DataParser.LoadDatasets();
            INeuralNetwork network = NetworkManager.NewNetwork(
                NetworkLayers.FullyConnected(784, 100, ActivationFunctionType.Sigmoid),
                NetworkLayers.FullyConnected(100, 10, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network, (training.X, training.Y), 60, 10, 0.5f, 0.5f,
                testParameters: new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Result.Cost}, accuracy: {p.Result.Accuracy}");
                })));
            Printf($"Stop reason: {result.StopReason}, elapsed time: {result.TrainingTime}");
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
