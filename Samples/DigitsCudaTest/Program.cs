using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Datasets;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp.PixelFormats;

namespace DigitsCudaTest
{
    public class Program
    {
        public static async Task Main()
        {
            // Create the network
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(28, 28),
                CuDnnNetworkLayers.Convolutional((5, 5), 20, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(ActivationType.LeakyReLU),
                CuDnnNetworkLayers.Convolutional((3, 3), 40, ActivationType.Identity),
                CuDnnNetworkLayers.Pooling(ActivationType.LeakyReLU),
                CuDnnNetworkLayers.FullyConnected(125, ActivationType.LeCunTanh),
                CuDnnNetworkLayers.Softmax(10));

            // Prepare the dataset
            ITrainingDataset trainingData = await Mnist.GetTrainingDatasetAsync(400); // Batches of 400 samples
            ITestDataset testData = await Mnist.GetTestDatasetAsync(p => Printf($"Epoch {p.Iteration}, cost: {p.Result.Cost}, accuracy: {p.Result.Accuracy}"));
            if (trainingData == null || testData == null)
            {
                Printf("Error downloading the datasets");
                Console.ReadKey();
                return;
            }

            // Setup and network training
            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cts.Cancel();
            TrainingSessionResult result = await NetworkManager.TrainNetworkAsync(network, 
                trainingData, 
                TrainingAlgorithms.AdaDelta(),
                20, 0.5f,
                TrackBatchProgress,
                testDataset: testData, token: cts.Token);

            // Save the training reports
            String
                timestamp = DateTime.Now.ToString("yy-MM-dd-hh-mm-ss"),
                path = Path.GetDirectoryName(Path.GetFullPath(Assembly.GetExecutingAssembly().Location)),
                dir = Path.Combine(path ?? throw new InvalidOperationException("The dll path can't be null"), "TrainingResults", timestamp);
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, $"{timestamp}_cost.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Cost));
            File.WriteAllText(Path.Combine(dir, $"{timestamp}_accuracy.py"), result.TestReports.AsPythonMatplotlibChart(TrainingReportType.Accuracy));
            network.Save(new FileInfo(Path.Combine(dir, $"{timestamp}{NetworkLoader.NetworkFileExtension}")));
            File.WriteAllText(Path.Combine(dir, $"{timestamp}.json"), network.SerializeMetadataAsJson());
            File.WriteAllText(Path.Combine(dir, $"{timestamp}_report.json"), result.SerializeAsJson());
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

        // Training monitor
        private static void TrackBatchProgress(BatchProgress progress)
        {
            Console.SetCursorPosition(0, Console.CursorTop);
            int n = (int)(progress.Percentage * 32 / 100); // 32 is the number of progress '=' characters to display
            char[] c = new char[32];
            for (int i = 0; i < 32; i++) c[i] = i <= n ? '=' : ' ';
            Console.Write($"[{new String(c)}] ");
        }
    }
}
