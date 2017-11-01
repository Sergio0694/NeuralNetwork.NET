using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using MnistDatasetToolkit;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
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

        public static async Task PerformBenchmarkAsync(LearningAlgorithmType type, bool convolution, int? samplesLimit, bool cacheEnabled, params int[] neurons)
        {
            Printf("Loading sample data");
            (double[,] dataset, double[,] y, double[,] test, double[,] yHat) = DataParser.ParseDataset(samplesLimit);

            double[,] x;
            if (convolution)
            {
                Printf("Preparing 2D data for the convolution layer");
                IReadOnlyList<double[,]> raw = dataset.Extract3DVolume();

                Printf("Processing through the convolution layer");
                IReadOnlyList<ConvolutionsStack> processed = SharedPipeline.Process(raw);
                x = ConvolutionPipeline.ConvertToMatrix(processed.ToArray());
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
                ? await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, y,
                    type, cts.Token, progress, neurons)
                : await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, y,
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
                Printf("Preparing test data");
                IReadOnlyList<double[,]> _2dTest = test.Extract3DVolume();

                Printf("Processing test data through the convolution layer");
                IReadOnlyList<ConvolutionsStack> convolutions = SharedPipeline.Process(_2dTest);
                xt = ConvolutionPipeline.ConvertToMatrix(convolutions.ToArray());
            }
            else xt = test;

            Printf("Calculating error");
            double[,] estimation = network.Forward(xt);
            int valid = 0;
            for (int i = 0; i < yHat.GetLength(0); i++)
            {
                // Extracts the index of the maximum valud
                int MaxIndex(double[] v)
                {
                    int index = 0;
                    double max = 0;
                    for (int j = 0; j < v.Length; j++)
                    {
                        if (v[j] > max)
                        {
                            max = v[j];
                            index = j;
                        }
                    }
                    return index;
                }

                // Iterate through every test sample
                double[] temp = new double[10];
                Buffer.BlockCopy(estimation, i * 10, temp, 0, sizeof(double) * 10);
                int estimationIndex = MaxIndex(temp);
                Buffer.BlockCopy(yHat, i * 10, temp, 0, sizeof(double) * 10);
                int expectedIndex = MaxIndex(temp);
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
            v => v.Expand(ConvolutionExtensions.Convolute3x3,
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
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2), // 26*26*10 >> 13*13*10
            v => v.Process(ConvolutionExtensions.Normalize),
            v => v.Expand(ConvolutionExtensions.Convolute3x3,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen),// 13*13*10 >> 11*11*60
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2), // 11*11*60 >> 5*5*60
            v => v.Process(ConvolutionExtensions.Normalize),
            v => v.Expand(ConvolutionExtensions.Convolute3x3,
                KernelsCollection.TopSobel,
                KernelsCollection.RightSobel,
                KernelsCollection.LeftSobel,
                KernelsCollection.BottomSobel,
                KernelsCollection.Outline,
                KernelsCollection.Sharpen,
                KernelsCollection.BottomRightEmboss,
                KernelsCollection.TopLeftEmboss), // 5*5*60 >> 3*3*480
            v => v.Process(ConvolutionExtensions.ReLU), // Set minimum threshold
            v => v.Process(ConvolutionExtensions.Pool2x2)); // 3*3*480 >> 1*1*480)); // Set minimum threshold);
    }
}
