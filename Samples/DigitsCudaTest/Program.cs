using System;
using MnistDatasetToolkit;
using NeuralNetworkNET.Cuda.APIs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace DigitsCudaTest
{
    class Program
    {
        static void Main()
        {
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            (var training, var test) = DataParser.LoadDatasets();
            var network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(120, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            network.StochasticGradientDescent((training.X, training.Y), 100, 10000, 
                null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Console.WriteLine($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
                })),
                0.5, 5);
            Console.ReadKey();
        }
    }
}
