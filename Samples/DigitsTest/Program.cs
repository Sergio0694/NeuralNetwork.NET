using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using MnistDatasetToolkit;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning;

namespace DigitsTest
{
    class Program
    {
        static async Task Main(String[] args)
        {
            if (args.Length != 2) args = new[] { @"C:\Users\Sergi\Documents\Digits", "100" };
            Printf("Loading sample data");
            (double[,] dataset, double[,] y) = DataParser.ParseDataset(
                $@"{args[0]}\train-images.idx3-ubyte", $@"{args[0]}\train-labels.idx1-ubyte", int.Parse(args[1]));
            Printf("Loading test data");
            (double[,] test, double[,] yHat) = DataParser.ParseDataset(
                $@"{args[0]}\t10k-images.idx3-ubyte", $@"{args[0]}\t10k-labels.idx1-ubyte");

            Printf("Preparing 2D data for the convolution layer");
            IReadOnlyList<double[,]> raw = DataParser.ConvertDatasetTo2dImages(dataset);

            Printf("Processing through the convolution layer");
            IReadOnlyList<ConvolutionsStack> processed = SharedPipeline.Process(raw);
            double[,] inputs = ConvolutionPipeline.ConvertToMatrix(processed.ToArray());

            // Get the optimized network
            Printf("Training");
            CancellationTokenSource cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            SingleLayerPerceptron network = await GradientDescentNetworkTrainer.ComputeTrainedNetworkAsync(inputs, y, 16, cts.Token, null,
                new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Iteration #{p.Iteration} >> {p.Cost}");
                }));

            Printf("Preparing test data");
            IReadOnlyList<double[,]> _2dTest = DataParser.ConvertDatasetTo2dImages(test);

            Printf("Processing test data through the convolution layer");
            IReadOnlyList<ConvolutionsStack> convolutions = SharedPipeline.Process(_2dTest);
            double[,] testMatrix = ConvolutionPipeline.ConvertToMatrix(convolutions.ToArray());

            Printf("Calculating error");
            double[,] estimation = network.Forward(testMatrix);
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
        private static void Printf([NotNull] String text)
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
