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

namespace DigitsRecognitionSample
{
    /// <summary>
    /// Logica di interazione per MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            this.MouseMove += (s, e) =>
            {
                if (e.LeftButton == MouseButtonState.Pressed) this.DragMove();
            };
        }

        private void OnTopToggle_OnOnToggled(object sender, EventArgs e) => Topmost = !Topmost;

        private void MinimizeButton_Clicked(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

        private void CloseButton_Clicked(object sender, RoutedEventArgs e) => Application.Current.Shutdown();

        #region Navigation

        /// <summary>
        /// Manages the navigation by making sure no double instances are created
        /// </summary>
        /// <typeparam name="T">The page to open</typeparam>
        private void NavigatePage<T>() where T : Window, new()
        {
            Window window = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.GetType() == typeof(T));
            if (window == null)
            {
                new T { Left = this.Left + 25, Top = this.Top + 25 }.Show();
            }
            else
            {
                window.Focus();
            }
        }

        #endregion

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog picker = new OpenFileDialog
            {
                DefaultExt = ".png",
                Filter = "PNG image (.png)|*.png"
            };
            bool? result = picker.ShowDialog();
            if (result.HasValue && result.Value)
            {
                Bitmap
                    image = new Bitmap(picker.FileName),
                    grayscale = image.ToGrayscale();
                SaveFileDialog saver = new SaveFileDialog
                {
                    DefaultExt = ".png",
                    Filter = "PNG image (.png)|*.png"
                };
                //result = saver.ShowDialog();
                //if (!(result.HasValue && result.Value)) return;
                //grayscale.Save(saver.FileName);

                // Example convolution pipeline
                double[,] normalized = grayscale.ToNormalizedPixelData();
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
                double[][,] volume = pipeline.Process(normalized);
            }
        }
    }
}
