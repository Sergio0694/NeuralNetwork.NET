using System;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.Cuda.APIs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace DigitsCudaTest
{
    class Program
    {
        static async Task Main()
        {
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Cpu;
            (var training, var test) = DataParser.LoadDatasets();
            var network = NeuralNetwork.NewRandom(
                NetworkLayer.Inputs(784),
                NetworkLayer.FullyConnected(30, ActivationFunctionType.Sigmoid),
                NetworkLayer.FullyConnected(10, ActivationFunctionType.Sigmoid));
            network.StochasticGradientDescent((training.X, training.Y), (test.X, test.Y), 20, 100, 3);
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
