using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning;
using NeuralNetworkNET.SupervisedLearning.Misc;

namespace ManualDigitsTest
{
    class Program
    {
        private static INeuralNetwork _Network;

        static async Task Main()
        {
            // Prepare the input data
            int[] count = new int[10];
            IEnumerable<double[,]>[] raw = new IEnumerable<double[,]>[10];
            for (int i = 0; i < 10; i++)
            {
                String[] files = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\TrainingSet\{i}");
                count[i] = files.Length;
                raw[i] =
                    from file in files
                    let image = new Bitmap(file)
                    let grayscale = image.ToGrayscale()
                    select grayscale.ToNormalizedPixelData();
            }
            IReadOnlyList<double[,]> source = raw.SelectMany(g => g).ToArray();

            // Prepare the results
            int samples = count.Sum();
            double[,] y = new double[samples, 10];
            for (int i = 0; i < samples; i++)
            {
                int sum = 0, j = 0;
                for (int z = 0; z < 10; z++)
                {
                    sum += count[z];
                    if (sum > i) break;
                    j++;
                }
                y[i, j] = 1.0;
            }

            Printf("Processing through the convolution layer");
            IReadOnlyList<ConvolutionsStack> processed = SharedPipeline.Process(source);
            double[,] inputs = ConvolutionPipeline.ConvertToMatrix(processed.ToArray());

            // Get the optimized network
            Printf("Training");
            CancellationTokenSource cts = new CancellationTokenSource();
            void CancelToken(object sender, ConsoleCancelEventArgs e)
            {
                cts.Cancel();
                Console.CancelKeyPress -= CancelToken;
            }
            Console.CancelKeyPress += CancelToken;
            _Network = await BackpropagationNetworkTrainer.ComputeTrainedNetworkAsync(inputs, y, LearningAlgorithmType.GradientDescend, cts.Token, null,
                new Progress<BackpropagationProgressEventArgs>(p =>
                {
                    Printf($"Iteration #{p.Iteration} >> {p.Cost}");
                }), 480, 32, 16, 10);

            // Test the network
            while (Console.ReadLine() is String input)
            {
                double[,] test = new Bitmap(input).ToGrayscale().ToNormalizedPixelData();
                ConvolutionsStack convolution = SharedPipeline.Process(test);
                double[] flat = convolution.Flatten();
                double[] yHat = _Network.Forward(flat);
                int index = 0;
                double max = 0;
                for (int i = 0; i < yHat.Length; i++)
                {
                    if (yHat[i] > max)
                    {
                        max = yHat[i];
                        index = i;
                    }
                }
                Printf($"The number is {index}");
            }
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
