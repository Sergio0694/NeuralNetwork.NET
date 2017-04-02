using System;
using System.Linq;
using System.IO;
using System.Windows;
using System.Drawing;
using System.Windows.Input;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;
using ConvolutionalNeuralNetworkLibrary;
using ConvolutionalNeuralNetworkLibrary.Convolution;
using ConvolutionalNeuralNetworkLibrary.ImageProcessing;
using JetBrains.Annotations;
using DigitsRecognitionSample.Views;

namespace DigitsRecognitionSample.Views
{
    /// <summary>
    /// Logica di interazione per XYWindow.xaml
    /// </summary>
    public partial class XYWindow : Window
    {
        public XYWindow()
        {
            InitializeComponent();
            this.MouseMove += (s, e) =>
            {
                if (e.LeftButton == MouseButtonState.Pressed) this.DragMove();
            };
        }

        private void OnTopToggle_OnOnToggled(object sender, EventArgs e) => Topmost = !Topmost;

        private void MinimizeButton_Clicked(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

        private void CloseButton_Clicked(object sender, RoutedEventArgs e) => Close();

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            // Get the filenames
            String[]
                x = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\Assets\Samples\XY\X"),
                y = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\Assets\Samples\XY\Y");

            // Get the images
            Bitmap[]
                xbmp = x.Select(f => new Bitmap(f).ToGrayscale()).ToArray(),
                ybmp = y.Select(f => new Bitmap(f).ToGrayscale()).ToArray();

            // Normalize the data
            double[][,]
                xn = xbmp.Select(b => b.ToNormalizedPixelData()).ToArray(),
                yn = ybmp.Select(b => b.ToNormalizedPixelData()).ToArray();

            // Setup the kernel pipeline
            ConvolutionPipeline pipeline = new ConvolutionPipeline(new VolumicProcessor[]
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

            // Get the results matrix (number of samples by output nodes)
            int 
                _10 = x.Length,
                results = _10 + y.Length;
            double[,] expectation = new double[results, 2];
            for (int i = 0; i < results; i++)
            {
                expectation[i, 0] = i < _10 ? 1.0 : 0.0;
                expectation[i, 1] = i < _10 ? 0.0 : 1.0;
            }

            IReadOnlyList<double[,]> source = xn.Concat(yn).ToArray();
            NeuralNetwork network = NetworkTrainer.ComputeTrainedNetwork(source, pipeline, expectation, 100, 
                new Progress<CNNOptimizationProgress>(p => System.Diagnostics.Debug.WriteLine($"#{p.Iteration} >> {p.Cost}")));
        }
    }
}
