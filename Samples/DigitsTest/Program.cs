using System;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.SupervisedLearning.Optimization.Progress;
using SixLabors.ImageSharp.PixelFormats;

namespace DigitsTest
{
    class Program
    {
        static async Task Main()
        {
            (var training, var test) = DataParser.LoadDatasets();
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.CreateImage<Alpha8>(28, 28),
                NetworkLayers.Convolutional((5, 5), 20, ActivationFunctionType.Identity),
                NetworkLayers.Pooling(ActivationFunctionType.LeakyReLU),
                NetworkLayers.FullyConnected(100, ActivationFunctionType.LeCunTanh),
                NetworkLayers.Softmax(10));
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network,
                DatasetLoader.Training(training, 100), 
                TrainingAlgorithms.Adadelta(), 
                60, 0.5f,
                new Progress<BatchProgress>(p =>
                {
                    Console.SetCursorPosition(0, Console.CursorTop);
                    int n = (int)(p.Percentage * 32 / 100);
                    char[] c = new char[32];
                    for (int i = 0; i < 32; i++) c[i] = i <= n ? '=' : ' ';
                    Console.Write($"[{new String(c)}] ");
                }),
                testDataset: DatasetLoader.Test(test, new Progress<TrainingProgressEventArgs>(p =>
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
