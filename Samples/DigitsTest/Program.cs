using System;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace DigitsTest
{
    class Program
    {
        static async Task Main()
        {
            (var training, var test) = DataParser.LoadDatasets();
            INeuralNetwork network = NetworkTrainer.NewNetwork(
                NetworkLayers.FullyConnected(784, 100, ActivationFunctionType.Sigmoid),
                NetworkLayers.FullyConnected(100, 10, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            TrainingSessionResult result = await NetworkTrainer.TrainNetworkAsync(network, (training.X, training.Y), 60, 10, null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
                })), 0.5f, 0.5f);
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
