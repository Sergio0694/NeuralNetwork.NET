using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Cuda.APIs;
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
            NeuralNetworkGpuPreferences.ProcessingMode = ProcessingMode.Gpu;
            (var training, var test) = DataParser.LoadDatasets();
            INeuralNetwork network = NetworkManager.NewNetwork(
                NetworkLayers.Convolutional((28, 28, 1), (5, 5), 10, ActivationFunctionType.Identity),
                NetworkLayers.Pooling((24, 24, 10), ActivationFunctionType.Tanh),
                NetworkLayers.FullyConnected(12 * 12 * 10, 100, ActivationFunctionType.Tanh),
                NetworkLayers.Softmax(100, 10));

            // Setup and start the training
            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cts.Cancel();
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network, (training.X, training.Y), 4, 400, null,
                new TestParameters(test, new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Epoch {p.Iteration}, cost: {p.Result.Cost}, accuracy: {p.Result.Accuracy}");
                })), 0.1f, 0.5f, 1, cts.Token);

            // Save the training reports
            Printf($"Stop reason: {result.StopReason}, elapsed time: {result.TrainingTime}");
            String
                code = Assembly.GetExecutingAssembly().Location,
                dll = Path.GetFullPath(code),
                root = Path.GetDirectoryName(dll),
                path = Path.Combine(root ?? throw new InvalidOperationException("The dll path can't be null"), "TrainingResults", DateTime.Now.ToString("yy-MM-dd"));
            Directory.CreateDirectory(path);
            File.WriteAllText(Path.Combine(path, $"{DateTime.Now:yy-MM-dd}_cost.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Cost));
            File.WriteAllText(Path.Combine(path, $"{DateTime.Now:yy-MM-dd}_accuracy.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Accuracy));
            network.Save(new DirectoryInfo(path), DateTime.Now.ToString("yy-MM-dd"));
            File.WriteAllText(Path.Combine(path, $"{DateTime.Now:yy-MM-dd}.json"), network.SerializeAsJson());
            File.WriteAllText(Path.Combine(path, $"{DateTime.Now:yy-MM-dd}_report.json"), result.SerializeAsJson());
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
