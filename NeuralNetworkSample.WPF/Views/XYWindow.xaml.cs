using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using JetBrains.Annotations;
using Microsoft.Win32;
using NeuralNetworkSampleWPF.Helpers;

namespace NeuralNetworkSampleWPF.Views
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

        private void CloseButton_Clicked(object sender, RoutedEventArgs e)
        {
            _Cts?.Cancel();
            Close();
        }

        [CanBeNull]
        private INeuralNetwork _Network;

        private CancellationTokenSource _Cts;

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            // Token setup
            CancellationTokenSource cts = new CancellationTokenSource();
            _Cts?.Cancel();
            _Cts = cts;

            // Get the filenames
            String[]
                x = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\Assets\Samples\XO\X"),
                o = Directory.GetFiles($@"{AppDomain.CurrentDomain.BaseDirectory}\Assets\Samples\XO\O");

            // Get the images
            Bitmap[]
                xbmp = x.Select(f => new Bitmap(f).ToGrayscale()).ToArray(),
                obmp = o.Select(f => new Bitmap(f).ToGrayscale()).ToArray();

            // Normalize the data
            double[][,]
                xn = xbmp.Select(b => b.ToNormalizedPixelData()).ToArray(),
                on = obmp.Select(b => b.ToNormalizedPixelData()).ToArray();

            // Get the results matrix (number of samples by output nodes)
            int
                _10 = x.Length,
                results = _10 + o.Length;
            double[,] expectation = new double[results, 2];
            for (int i = 0; i < results; i++)
            {
                expectation[i, 0] = i < _10 ? 1.0 : 0.0;
                expectation[i, 1] = i < _10 ? 0.0 : 1.0;
            }

            // Convolution
            IReadOnlyList<double[,]> source = xn.Concat(on).ToArray();
            IReadOnlyList<ConvolutionsStack> convolutions = App.SharedPipeline.Process(source);
            double[,] inputs = ConvolutionPipeline.ConvertToMatrix(convolutions.ToArray());

            // Training
            _Network = await GradientDescentNetworkTrainer.ComputeTrainedNetworkAsync(inputs, expectation, 200, cts.Token, null,
            new Progress<BackpropagationProgressEventArgs>(p =>
            {
                Dispatcher.BeginInvoke(DispatcherPriority.Normal, new Action(() =>
                {
                    GenBlock.Text = $"#{p.Iteration}";
                    ValueBlock.Text = p.Cost.ToString(CultureInfo.InvariantCulture);
                }));
            }));
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            // Token cancellation
            _Cts?.Cancel();

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
                ConvolutionsStack data = App.SharedPipeline.Process(normalized);
                double[] flat = data.Flatten();
                double[] yHat = _Network.Forward(flat);
                int index = yHat[0] > yHat[1] ? 0 : 1;
                MessageBox.Show($"The symbol is {(index == 0 ? "X" : "O")}");
            }
        }
    }
}
