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

        public static async Task PerformBenchmarkAsync(LearningAlgorithmType type, int? batchSize, bool convolution, int? samplesLimit, bool cacheEnabled, params NetworkLayer[] layers)
        {
            Printf("Loading sample data");
            (var trainingSet, var testSet) = DataParser.LoadDatasets();

            float[,] x;
            if (convolution)
            {
                Printf("Processing through the convolution layer");
                x = SharedPipeline.Process(trainingSet.X);
            }
            else x = trainingSet.X;

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
                ? await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, trainingSet.Y, batchSize,
                    type, cts.Token, progress, layers)
                : await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(x, trainingSet.Y, batchSize,
                    previous, type, cts.Token, progress);

            if (cacheEnabled)
            {
                Printf("Storing computed network");
                using (StreamWriter stream = File.CreateText(Path.Combine(ExecutingPath, "MNIST", "Network.json")))
                {
                    stream.Write(network.SerializeAsJSON());
                }
            }

            float[,] xt;
            if (convolution)
            {
                Printf("Processing test data through the convolution layer");
                xt = SharedPipeline.Process(testSet.X);
            }
            else xt = testSet.X;

            Printf("Calculating error");
            float[,] estimation = network.Forward(xt);
            int valid = 0;
            for (int i = 0; i < 10_000; i++)
            {
                // Iterate through every test sample
                float[] temp = new float[10];
                Buffer.BlockCopy(estimation, i * 10, temp, 0, sizeof(float) * 10);
                int estimationIndex = temp.Argmax();
                Buffer.BlockCopy(testSet.Y, i * 10, temp, 0, sizeof(float) * 10);
                int expectedIndex = temp.Argmax();
                if (estimationIndex == expectedIndex) valid++;
            }
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("\n=======================\n");
            Printf($"{valid}/10000, {(float)valid / 10000 * 100:###.##}%");
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
            ConvolutionOperation.Pool2x2); // 3*3*480 >> 1*1*480)); // Set minimum threshold
    }
}
