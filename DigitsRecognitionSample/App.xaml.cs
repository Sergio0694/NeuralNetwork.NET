using System.Data;
using System.Linq;
using System.Windows;
using ConvolutionalNeuralNetworkLibrary;
using ConvolutionalNeuralNetworkLibrary.Convolution;

namespace DigitsRecognitionSample
{
    /// <summary>
    /// Logica di interazione per App.xaml
    /// </summary>
    public partial class App : Application
    {
        /// <summary>
        /// Gets the shared convolution pipeline that takes a 28*28 image and returns 480 node values
        /// </summary>
        public static ConvolutionPipeline SharedPipeline { get; } = new ConvolutionPipeline(new VolumicProcessor[]
        {
            // 10 kernels, 28*28*1 pixels >> 26*26*10
            v => new double[][,]
            {
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopSobel),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.RightSobel),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.LeftSobel),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomSobel),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.Outline),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.Sharpen),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomLeftEmboss),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopRightEmboss),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopLeftEmboss),
                MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomRightEmboss)
            },
            v => v.Select(MatrixHelper.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(MatrixHelper.Pool2x2).ToArray(), // 26*26*10 >> 13*13*10
            v => v.Select(MatrixHelper.Normalize).ToArray(),
            v => v.Select(feature =>
            {
                return new double[][,]
                {
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.RightSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.LeftSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.Outline),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.Sharpen),
                };
            }).SelectMany(group => group).ToArray(), // 13*13*10 >> 11*11*60
            v => v.Select(MatrixHelper.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(MatrixHelper.Pool2x2).ToArray(), // 11*11*60 >> 5*5*60
            v => v.Select(MatrixHelper.Normalize).ToArray(),
            v => v.Select(feature =>
            {
                return new double[][,]
                {
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.RightSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.LeftSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomSobel),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.Outline),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.Sharpen),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.BottomRightEmboss),
                    MatrixHelper.Convolute3x3(v[0], KernelsCollection.TopLeftEmboss)
                };
            }).SelectMany(group => group).ToArray(), // 5*5*60 >> 3*3*480
            v => v.Select(MatrixHelper.ReLU).ToArray(), // Set minimum threshold
            v => v.Select(MatrixHelper.Pool2x2).ToArray() // 3*3*360 >> 1*1*480
        });
    }
}
