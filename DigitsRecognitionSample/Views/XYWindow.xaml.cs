﻿using System;
using System.Linq;
using System.IO;
using System.Windows;
using System.Drawing;
using System.Windows.Input;
using System.Collections.Generic;
using System.Windows.Threading;
using System.Threading;
using System.Threading.Tasks;
using ConvolutionalNeuralNetworkLibrary;
using ConvolutionalNeuralNetworkLibrary.ImageProcessing;
using JetBrains.Annotations;
using Microsoft.Win32;

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

        [CanBeNull]
        private NeuralNetwork _Network;

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
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

            IReadOnlyList<double[,]> source = xn.Concat(on).ToArray();
            _Network = await Task.Run(() =>
            {
                return NetworkTrainer.ComputeTrainedNetwork(source, App.SharedPipeline, expectation, 200, CancellationToken.None, null,
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
                int index = yHat[0] > yHat[1] ? 0 : 1;
                MessageBox.Show($"The symbol is {(index == 0 ? "X" : "O")}");
            }
        }
    }
}