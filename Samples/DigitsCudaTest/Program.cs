using System;
using System.Threading.Tasks;
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
        static async Task Main()
        {
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            (var training, var test) = DataParser.LoadDatasets();
            var network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(100, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            network.StochasticGradientDescent((training.X, training.Y), 20, 100, 
                null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Console.WriteLine($"Epoch {p.Iteration}, cost: {p.Cost}, accuracy: {p.Accuracy}");
                })),
                0.5, 5);
            Console.ReadKey();
            /*
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            await MnistTester.PerformBenchmarkAsync(LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence, 1000, false, null, false, 
                NetworkLayer.Inputs(784), 
                NetworkLayer.FullyConnected(16, ActivationFunctionType.Sigmoid), 
                NetworkLayer.FullyConnected(16, ActivationFunctionType.Sigmoid), 
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid)); */
        }
    }
}
