using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using DigitsRecognitionSample.Helpers;
using NeuralNetworkNET;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Convolution.Misc;

namespace DigitsRecognitionSample
{
    /// <summary>
    /// Logica di interazione per App.xaml
    /// </summary>
    public partial class App : Application
    {
        // Library initialization
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            ParallelCompatibilityWrapper.Initialize((start, end, body) => Parallel.For(start, end, body).IsCompleted);
            AccordNETGradientOptimizationMethodCompatibilityWrapper.Initialize(
                (variables, function, gradient) => new LBFGSWrapper(variables, function, gradient));
        }

        /// <summary>
        /// Gets the shared convolution pipeline that takes a 28*28 image and returns 480 node values
        /// </summary>
        public static ConvolutionPipeline SharedPipeline { get; } = new ConvolutionPipeline(

            // 10 kernels, 28*28*1 pixels >> 26*26*10
            v => new[]
            {
                v[0].Convolute3x3(KernelsCollection.TopSobel),
                v[0].Convolute3x3(KernelsCollection.RightSobel),
                v[0].Convolute3x3(KernelsCollection.LeftSobel),
                v[0].Convolute3x3(KernelsCollection.BottomSobel),
                v[0].Convolute3x3(KernelsCollection.Outline),
                v[0].Convolute3x3(KernelsCollection.Sharpen),
                v[0].Convolute3x3(KernelsCollection.BottomLeftEmboss),
                v[0].Convolute3x3(KernelsCollection.TopRightEmboss),
                v[0].Convolute3x3(KernelsCollection.TopLeftEmboss),
                v[0].Convolute3x3(KernelsCollection.BottomRightEmboss)
            },
            v => v.Select(ConvolutionExtensions.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(ConvolutionExtensions.Pool2x2).ToArray(), // 26*26*10 >> 13*13*10
            v => v.Select(ConvolutionExtensions.Normalize).ToArray(),
            v => v.Select(feature => new[]
            {
                feature.Convolute3x3(KernelsCollection.TopSobel),
                feature.Convolute3x3(KernelsCollection.RightSobel),
                feature.Convolute3x3(KernelsCollection.LeftSobel),
                feature.Convolute3x3(KernelsCollection.BottomSobel),
                feature.Convolute3x3(KernelsCollection.Outline),
                feature.Convolute3x3(KernelsCollection.Sharpen),
            }).ToArray(), // 13*13*10 >> 11*11*60
            v => v.Select(ConvolutionExtensions.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(ConvolutionExtensions.Pool2x2).ToArray(), // 11*11*60 >> 5*5*60
            v => v.Select(ConvolutionExtensions.Normalize).ToArray(),
            v => v.Select(feature => new[]
            {
                feature.Convolute3x3(KernelsCollection.TopSobel),
                feature.Convolute3x3(KernelsCollection.RightSobel),
                feature.Convolute3x3(KernelsCollection.LeftSobel),
                feature.Convolute3x3(KernelsCollection.BottomSobel),
                feature.Convolute3x3(KernelsCollection.Outline),
                feature.Convolute3x3(KernelsCollection.Sharpen),
                feature.Convolute3x3(KernelsCollection.BottomRightEmboss),
                feature.Convolute3x3(KernelsCollection.TopLeftEmboss)
            }).ToArray(), // 5*5*60 >> 3*3*480
            v => v.Select(ConvolutionExtensions.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(ConvolutionExtensions.Pool2x2).ToArray()); // 3*3*480 >> 1*1*480)); // Set minimum threshold);
    }
}
