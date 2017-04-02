using System;
using System.Linq;
using System.IO;
using System.Windows;
using System.Drawing;
using System.Windows.Input;
using System.Collections.Generic;
using System.Windows.Threading;
using System.Threading.Tasks;
using ConvolutionalNeuralNetworkLibrary;
using System.Threading;
using ConvolutionalNeuralNetworkLibrary.Convolution;
using ConvolutionalNeuralNetworkLibrary.ImageProcessing;
using JetBrains.Annotations;
using Microsoft.Win32;

namespace DigitsRecognitionSample.Views
{
    /// <summary>
    /// Logica di interazione per DigitsWindow.xaml
    /// </summary>
    public partial class DigitsWindow : Window
    {
        public DigitsWindow()
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

        [CanBeNull]
        private NeuralNetwork _Network;

        private CancellationTokenSource _Cts;

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            // Token setup
            CancellationTokenSource cts = new CancellationTokenSource();
            _Cts?.Cancel();
            _Cts = cts;

            // Prepare the input data
            int[] count = new int[10];
            IEnumerable<double[,]>[] raw = new IEnumerable<double[,]>[10];
            for (int i = 0; i < 10; i++)
            {
                String[] files = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\Assets\Samples\Digits\{i}");
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

            // Get the optimized network
            double[] solution = _Network?.SerializeWeights();
            _Network = await Task.Run(() =>
            {
                return NetworkTrainer.ComputeTrainedNetwork(source, App.SharedPipeline, y, 400, cts.Token, solution,
                    new Progress<CNNOptimizationProgress>(p =>
                    {
                        Dispatcher.BeginInvoke(DispatcherPriority.Normal, new Action(() =>
                        {
                            GenBlock.Text = $"#{p.Iteration}";
                            ValueBlock.Text = p.Cost.ToString();
                        }));
                    }));
            });
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            // Token cancellation
            _Cts?.Cancel();

            // Handle the user input
            if (_Network == null || App.SharedPipeline == null) return;
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
                double[,] normalized = grayscale.ToNormalizedPixelData();
                double[][,] data = App.SharedPipeline.Process(normalized);
                double[] flat = data.Flatten();
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
                MessageBox.Show($"The number is {index}");
            }
        }
    }
}
