using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Convolution.Operations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace SharedBenchmark
{
    /// <summary>
    /// A simple class that exposes shared code to test the neural network trainer on the MNIST dataset
    /// </summary>
    public static class MnistTester
    {
        /// <summary>
        /// Gets the path of the executing dll
        /// </summary>
        private static String ExecutingPath
        {
            get
            {
                String
                    code = Assembly.GetExecutingAssembly().Location,
                    dll = Path.GetFullPath(code),
                    root = Path.GetDirectoryName(dll);
                return root;
            }
        }

        public static async Task PerformBenchmarkAsync(LearningAlgorithmType type, NeuralNetworkType networkType, int? batchSize, bool convolution, int? samplesLimit, bool cacheEnabled, params int[] neurons)
        {
            Printf("Loading sample data");
            (double[,] dataset, double[,] y, double[,] test, double[,] yHat) = DataParser.ParseDataset(samplesLimit);

            double[,] x;
            if (convolution)
            {
                Printf("Processing through the convolution layer");
                x = SharedPipeline.Process(dataset);
            }
            else x = dataset;

            INeuralNetwork previous;
            if (cacheEnabled)
            {
                Printf("Retrieving previous network");
                try
                {
                    String json = File.ReadAllText(Path.Combine(ExecutingPath, "MNIST", "Network.json"));
                    previous = NeuralNetworkDeserializer.TryDeserialize(json);
                }
                catch (FileNotFoundException)
                {
                    // Skip!
                    Printf("Previous network not found");
                    previous = null;
                }
            }
            else previous = null;

            // Get the optimized network
            Printf("Training");
            CancellationTokenSource cts = new CancellationTokenSource();
            void CancelToken(object sender, ConsoleCancelEventArgs e)
            {
                cts.Cancel();
                Console.CancelKeyPress -= CancelToken;
            }
            Console.CancelKeyPress += CancelToken;
            IProgress<BackpropagationProgressEventArgs> progress = new Progress<BackpropagationProgressEventArgs>(p =>
            {
                Printf($"Iteration #{p.Iteration} >> {p.Cost}");
            });
            INeuralNetwork network = previous == null
                ? await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, y, batchSize,
                    type, networkType, cts.Token, progress, neurons)
                : await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, y, batchSize,
                    previous, type, cts.Token, progress);

            if (cacheEnabled)
            {
                Printf("Storing computed network");
                using (StreamWriter stream = File.CreateText(Path.Combine(ExecutingPath, "MNIST", "Network.json")))
                {
                    stream.Write(network.SerializeAsJSON());
                }
            }

            double[,] xt;
            if (convolution)
            {
                Printf("Processing test data through the convolution layer");
                xt = SharedPipeline.Process(test);
            }
            else xt = test;

            Printf("Calculating error");
            double[,] estimation = network.Forward(xt);
            int valid = 0;
            for (int i = 0; i < yHat.GetLength(0); i++)
            {
                // Iterate through every test sample
                double[] temp = new double[10];
                Buffer.BlockCopy(estimation, i * 10, temp, 0, sizeof(double) * 10);
                int estimationIndex = temp.IndexOfMax();
                Buffer.BlockCopy(yHat, i * 10, temp, 0, sizeof(double) * 10);
                int expectedIndex = temp.IndexOfMax();
                if (estimationIndex == expectedIndex) valid++;
            }
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("\n=======================\n");
            Printf($"{valid}/{yHat.GetLength(0)}, {(double)valid / yHat.GetLength(0) * 100:###.##}%");
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

        /// <summary>
        /// Gets the shared convolution pipeline that takes a 28*28 image and returns 480 node values
        /// </summary>
        public static ConvolutionPipeline SharedPipeline { get; } = new ConvolutionPipeline(

            // 10 kernels, 28*28*1 pixels >> 26*26*10
            ConvolutionOperation.Convolution3x3(
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomLeftEmboss,
                KernelsCollection.TopRightEmboss,
                KernelsCollection.TopLeftEmboss,
                KernelsCollection.BottomRightEmboss),
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2, // 26*26*10 >> 13*13*10
            ConvolutionOperation.Convolution3x3(
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen),// 13*13*10 >> 11*11*60
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2, // 11*11*60 >> 5*5*60
            ConvolutionOperation.Convolution3x3(
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomRightEmboss,
                KernelsCollection.TopLeftEmboss), // 5*5*60 >> 3*3*480
            ConvolutionOperation.ReLU, // Set minimum threshold
            ConvolutionOperation.Pool2x2,
            ConvolutionOperation.Pool2x2); // 3*3*480 >> 1*1*480)); // Set minimum threshold);

#if NET47
        public static void ExtractDatasetImages(String path)
        {
            (double[,] x, double[,] xlabels, double[,] y, double[,] ylabels) = DataParser.ParseDataset();
            String
                root = Path.Combine(path, "MNIST_images"),
                xPath = Path.Combine(root, "X"),
                yPath = Path.Combine(root, "Y");
            Directory.CreateDirectory(root);
            Directory.CreateDirectory(xPath);
            Directory.CreateDirectory(yPath);
            int
                hx = x.GetLength(0),
                hy = y.GetLength(0);
            void ExtractImages(double[,] data, double[,] dataLabels, int index, String dir)
            {
                double[] label = new double[10];
                Buffer.BlockCopy(dataLabels, sizeof(double) * 10 * index, label, 0, sizeof(double) * 10);
                double[] xyValue = new double[784];
                Buffer.BlockCopy(data, sizeof(double) * 784 * index, xyValue, 0, sizeof(double) * 784);
                int number = label.IndexOfMax();
                String valuePath = Path.Combine(dir, number.ToString());
                Directory.CreateDirectory(valuePath);
                System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap(28, 28);
                for (int j = 0; j < 28; j++)
                    for (int k = 0; k < 28; k++)
                    {
                        int offset = j * 28 + k;
                        double color = xyValue[offset];
                        int normalized = (int)((1d - color) * 255d);
                        bitmap.SetPixel(k, j, System.Drawing.Color.FromArgb(normalized, normalized, normalized));
                    }
                bitmap.Save(Path.Combine(valuePath, $"{index}.jpg"));
            }
            Parallel.For(0, hx, i =>
            {
                ExtractImages(x, xlabels, i, xPath);
                if (i % 1000 == 0) Console.WriteLine($"X - {i}");
            });
            Parallel.For(0, hy, i =>
            {
                ExtractImages(y, ylabels, i, yPath);
                if (i % 1000 == 0) Console.WriteLine($"Y - {i}");
            });
            Console.ReadKey();
        }
#endif
    }
}
