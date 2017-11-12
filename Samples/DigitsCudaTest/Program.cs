using System;
using MnistDatasetToolkit;
using NeuralNetworkNET.Cuda.APIs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace DigitsCudaTest
{
    class Program
    {
        static void Main()
        {
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Cpu;
            (var training, var test) = DataParser.LoadDatasets();
            NeuralNetwork network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(100, ActivationFunctionType.Sigmoid),
                NetworkLayer.Outputs(10, ActivationFunctionType.Sigmoid, CostFunctionType.CrossEntropy));
            network.StochasticGradientDescent((training.X, training.Y), 60, 10, 
                null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
                })),
                0.1f, 5f);
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
