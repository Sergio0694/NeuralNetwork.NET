using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading;
using Windows.ApplicationModel;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Storage;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using JetBrains.Annotations;
using NeuralNetworkNET.Convolution;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning;

namespace NeuralNetworkSampleUWP
{
    /// <summary>
    /// Pagina vuota che può essere usata autonomamente oppure per l'esplorazione all'interno di un frame.
    /// </summary>
    public sealed partial class DigitsPage : Page
    {
        public DigitsPage()
        {
            this.InitializeComponent();
        }

        [CanBeNull]
        private INeuralNetwork _Network;

        private CancellationTokenSource _Cts;

        private async void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            /*
            // Token setup
            CancellationTokenSource cts = new CancellationTokenSource();
            _Cts?.Cancel();
            _Cts = cts;

            // Prepare the input data
            int[] count = new int[10];
            IEnumerable<double[,]>[] raw = new IEnumerable<double[,]>[10];
            String path = $@"{Package.Current.InstalledLocation.Path}\Assets\Samples";
            for (int i = 0; i < 10; i++)
            {
                StorageFolder folder = await StorageFolder.GetFolderFromPathAsync($@"path\{i}");
                IReadOnlyList<StorageFile> files = await folder.GetFilesAsync();
                count[i] = files.Count;

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

            // Convolution
            IReadOnlyList<ConvolutionsStack> convolutions = App.SharedPipeline.Process(source);
            double[,] inputs = ConvolutionPipeline.ConvertToMatrix(convolutions.ToArray());

            // Get the optimized network
            _Network = await GradientDescentNetworkTrainer.ComputeTrainedNetworkAsync(inputs, y, 90, cts.Token, null,
            new Progress<BackpropagationProgressEventArgs>(p =>
            {
                Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
                {
                    GenBlock.Text = $"#{p.Iteration}";
                    ErrorBlock.Text = p.Cost.ToString();
                });
            }));

    */
        }
    }
}
