using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace DigitsCudaTest
{
    class Program
    {
        static async Task Main()
        {
            // Parse the dataset and create the network
            (var training, var test) = DataParser.LoadDatasets();
            INeuralNetwork network = NetworkManager.NewNetwork(
                CuDnnNetworkLayers.Convolutional((28, 28, 1), (5, 5), 20, ActivationFunctionType.LeakyReLU),
                CuDnnNetworkLayers.Convolutional((24, 24, 20), (5, 5), 20, ActivationFunctionType.Identity),
                CuDnnNetworkLayers.Pooling((20, 20, 20), ActivationFunctionType.LeakyReLU),
                CuDnnNetworkLayers.Convolutional((10, 10, 20), (3, 3), 40, ActivationFunctionType.LeakyReLU),
                CuDnnNetworkLayers.Convolutional((8, 8, 40), (3, 3), 40, ActivationFunctionType.Identity),
                CuDnnNetworkLayers.Pooling((6, 6, 40), ActivationFunctionType.LeakyReLU),
                CuDnnNetworkLayers.FullyConnected(3 * 3 * 40, 125, ActivationFunctionType.LeCunTanh),
                CuDnnNetworkLayers.FullyConnected(125, 64, ActivationFunctionType.LeCunTanh),
                CuDnnNetworkLayers.Softmax(64, 10));

            // Setup and start the training
            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cts.Cancel();
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network, (training.X, training.Y), 60, 400,
                TrainingAlgorithmsInfo.CreateForAdadelta(), 0.5f,
                testParameters: new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Result.Cost}, accuracy: {p.Result.Accuracy}");
                })), token: cts.Token);

            // Save the training reports
            Printf($"Stop reason: {result.StopReason}, elapsed time: {result.TrainingTime}");
            String
                root = Path.GetDirectoryName(Path.GetFullPath(Assembly.GetExecutingAssembly().Location)),
                path = Path.Combine(root ?? throw new InvalidOperationException("The dll path can't be null"), "TrainingResults", DateTime.Now.ToString("yy-MM-dd"));
            Directory.CreateDirectory(path);
            String timestamp = DateTime.Now.ToString("yy-MM-dd-mm-ss");
            File.WriteAllText(Path.Combine(path, $"{timestamp}_cost.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Cost));
            File.WriteAllText(Path.Combine(path, $"{timestamp}_accuracy.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Accuracy));
            network.Save(new DirectoryInfo(path), timestamp);
            File.WriteAllText(Path.Combine(path, $"{timestamp}.json"), network.SerializeAsJson());
            File.WriteAllText(Path.Combine(path, $"{timestamp}_report.json"), result.SerializeAsJson());
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
