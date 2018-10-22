using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.APIs.Datasets
{
    /// <summary>
    /// A static class that provides quick access to the MNIST database, see <a href="http://yann.lecun.com/exdb/mnist/">yann.lecun.com/exdb/mnist/</a>
    /// </summary>
    public static class Mnist
    {
        #region Constants

        // The training samples in the train-images-idx3-ubyte.gz file
        private const int TrainingSamples = 60000;

        // The test samples in the t10k-labels-idx1-ubyte.gz file
        private const int TestSamples = 10000;

        private const int SampleSize = 784;
        
        private const string MnistHttpRootPath = "http://yann.lecun.com/exdb/mnist/";
        
        private const string TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";
        
        private const string TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";
        
        private const string TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";
        
        private const string TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";

        #endregion

        /// <summary>
        /// Downloads the MNIST training datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITrainingDataset> GetTrainingDatasetAsync(int size, CancellationToken token = default)
        {
            Func<Stream>[] factories = await Task.WhenAll(
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TrainingSetValuesFilename}", null, token), 
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TrainingSetLabelsFilename}", null, token));
            if (factories.Any(s => s == null)) return null;
            (float[,] X, float[,] Y) data = ParseSamples((factories[0], factories[1]), TrainingSamples);
            return data.X == null || data.Y == null
                ? null
                : DatasetLoader.Training(data, size);
        }

        /// <summary>
        /// Downloads the MNIST test datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITestDataset> GetTestDatasetAsync([CanBeNull] Action<TrainingProgressEventArgs> progress = null, CancellationToken token = default)
        {
            Func<Stream>[] factories = await Task.WhenAll(
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TestSetValuesFilename}", null, token), 
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TestSetLabelsFilename}", null, token));
            if (factories.Any(s => s == null)) return null;
            (float[,] X, float[,] Y) data = ParseSamples((factories[0], factories[1]), TestSamples);
            return data.X == null || data.Y == null
                ? null
                : DatasetLoader.Test(data, progress);
        }

        /// <summary>
        /// Downloads and exports the full MNIST dataset (both training and test samples) to the target directory
        /// </summary>
        /// <param name="directory">The target directory</param>
        /// <param name="token">The cancellation token for the operation</param>
        [PublicAPI]
        public static async Task<bool> ExportDatasetAsync([NotNull] DirectoryInfo directory, CancellationToken token = default)
        {
            Func<Stream>[] factories = await Task.WhenAll(
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TrainingSetValuesFilename}", null, token), 
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TrainingSetLabelsFilename}", null, token),
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TestSetValuesFilename}", null, token), 
                DatasetsDownloader.GetFileAsync($"{MnistHttpRootPath}{TestSetLabelsFilename}", null, token));
            if (factories.Any(s => s == null) || token.IsCancellationRequested) return false;
            if (!directory.Exists) directory.Create();
            ParallelLoopResult result = Parallel.ForEach(new (string Name, Func<Stream> X, Func<Stream> Y, int Count)[]
            {
                (TrainingSetValuesFilename, factories[0], factories[1], TrainingSamples),
                (TestSetValuesFilename, factories[2], factories[3], TestSamples)
            }, (tuple, state) =>
            {
                ExportSamples(directory, (tuple.Name, tuple.X, tuple.Y), tuple.Count, token);
                if (token.IsCancellationRequested) state.Stop();
            });
            return result.IsCompleted && !token.IsCancellationRequested;
        }

        #region Tools

        /// <summary>
        /// Parses a MNIST dataset
        /// </summary>
        /// <param name="factory">A pair of factories for the input <see cref="Stream"/> instances to read</param>
        /// <param name="count">The number of samples to parse</param>
        private static unsafe (float[,], float[,]) ParseSamples((Func<Stream> X, Func<Stream> Y) factory, int count)
        {
            // Input checks
            using (Stream inputs = factory.X(), labels = factory.Y())
            using (GZipStream
                xGzip = new GZipStream(inputs, CompressionMode.Decompress),
                yGzip = new GZipStream(labels, CompressionMode.Decompress))
            {
                float[,]
                    x = new float[count, SampleSize],
                    y = new float[count, 10];
                xGzip.Read(new byte[16], 0, 16);
                yGzip.Read(new byte[8], 0, 8);
                byte[] temp = new byte[SampleSize];
                fixed (float* px = x, py = y)
                fixed (byte* ptemp = temp)
                {
                    for (int i = 0; i < count; i++)
                    {
                        // Read the image pixel values
                        xGzip.Read(temp, 0, SampleSize);
                        int offset = i * SampleSize;
                        for (int j = 0; j < SampleSize; j++)
                            px[offset + j] = ptemp[j] / 255f;

                        // Read the label
                        py[i * 10 + yGzip.ReadByte()] = 1;
                    }
                    return (x, y);
                }
            }
        }

        /// <summary>
        /// Exports a MNIST dataset file
        /// </summary>
        /// <param name="folder">The target folder to use to save the images</param>
        /// <param name="source">A pair of factories for the input <see cref="Stream"/> instances to read</param>
        /// <param name="count">The number of samples to parse</param>
        /// <param name="token">A token for the operation</param>
        private static unsafe void ExportSamples([NotNull] DirectoryInfo folder, (string Name, Func<Stream> X, Func<Stream> Y) source, int count, CancellationToken token)
        {
            using (Stream inputs = source.X(), labels = source.Y())
            using (GZipStream
                xGzip = new GZipStream(inputs, CompressionMode.Decompress),
                yGzip = new GZipStream(labels, CompressionMode.Decompress))
            {
                xGzip.Read(new byte[16], 0, 16);
                yGzip.Read(new byte[8], 0, 8);
                byte[] temp = new byte[SampleSize];
                fixed (byte* ptemp = temp)
                {
                    if (token.IsCancellationRequested) return;
                    for (int i = 0; i < count; i++)
                    {
                        // Read the image pixel values
                        xGzip.Read(temp, 0, SampleSize);
                        int label = yGzip.ReadByte();
                        using (Image<Rgb24> image = new Image<Rgb24>(28, 28))
                            fixed (Rgb24* p0 = image.GetPixelSpan())
                            {
                                for (int j = 0; j < SampleSize; j++)
                                    p0[j] = new Rgb24(ptemp[j], ptemp[j], ptemp[j]);
                                using (FileStream file = File.OpenWrite(Path.Combine(folder.FullName, $"[{source.Name}][{i}][{label}].bmp")))
                                    image.SaveAsBmp(file);
                            }
                    }
                }
            }
        }

        #endregion
    }
}
