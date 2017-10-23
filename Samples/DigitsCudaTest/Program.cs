using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using MnistDatasetToolkit;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning;

namespace DigitsCudaTest
{
    class Program
    {
        private static void Test()
        {
            Random random = new Random(DateTime.Now.Millisecond);
            int x = 2000, y = 1500, z = 800;
            double[,]
                a = random.NextMatrix(x, y),
                b = random.NextMatrix(y, z);

            Stopwatch timer = new Stopwatch();
            timer.Start();
            var c1 = new double[x, z];
            int m = a.GetLength(0);
            int n = a.GetLength(1);
            int p = b.GetLength(1);
            
            Gpu.Default.For(0, m * p, index =>
            {
                int
                    i = index / p,
                    j = index % p;

                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                c1[i, j] = sum;
            });

            timer.Stop();
            var t1 = timer.Elapsed;
            timer.Restart();
            var c2 = a.Multiply(b);
            timer.Stop();
            var t2 = timer.Elapsed;
            Debug.Assert(c1.ContentEquals(c2));
            Debug.WriteLine($"{t1.TotalMilliseconds} vs {t2.TotalMilliseconds}");
        }

        static async Task Main(String[] args)
        {
            if (args.Length != 2) args = new[] { @"C:\Users\Sergi\Documents\Digits", "100" };
            Printf("Loading sample data");
            (double[,] dataset, double[,] y, double[,] test, double[,] yHat) = DataParser.ParseDataset(int.Parse(args[1]));

            Printf("Preparing 2D data for the convolution layer");
            IReadOnlyList<double[,]> raw = DataParser.ConvertDatasetTo2dImages(dataset);

            Printf("Processing through the convolution layer");
            IReadOnlyList<ConvolutionsStack> processed = SharedPipeline.Process(raw);
            double[,] inputs = ConvolutionPipeline.ConvertToMatrix(processed.ToArray());

            // Get the optimized network
            Printf("Training");
            CancellationTokenSource cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            INeuralNetwork network = await GradientDescentNetworkTrainer.ComputeTrainedNetworkAsync(inputs, y, LearningAlgorithmType.GradientDescend, cts.Token, null,
                new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Iteration #{p.Iteration} >> {p.Cost}");
                }), 480, 48, 32, 10);

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
