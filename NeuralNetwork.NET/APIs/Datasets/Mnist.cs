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

        private const String MnistHttpRootPath = "http://yann.lecun.com/exdb/mnist/";

        private const String TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";

        private const String TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";

        private const String TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";

        private const String TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";

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
                DatasetsDownloader.GetAsync($"{MnistHttpRootPath}{TrainingSetValuesFilename}", token), 
                DatasetsDownloader.GetAsync($"{MnistHttpRootPath}{TrainingSetLabelsFilename}", token));
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
        public static async Task<ITestDataset> GetTestDatasetAsync([CanBeNull] IProgress<TrainingProgressEventArgs> progress = null, CancellationToken token = default)
        {
            Func<Stream>[] factories = await Task.WhenAll(
                DatasetsDownloader.GetAsync($"{MnistHttpRootPath}{TestSetValuesFilename}", token), 
                DatasetsDownloader.GetAsync($"{MnistHttpRootPath}{TestSetLabelsFilename}", token));
            if (factories.Any(s => s == null)) return null;
            (float[,] X, float[,] Y) data = ParseSamples((factories[0], factories[1]), TestSamples);
            return data.X == null || data.Y == null
                ? null
                : DatasetLoader.Test(data, progress);
        }

        #region Tools

        /// <summary>
        /// Parses a MNIST dataset
        /// </summary>
        /// <param name="factory">A pair of factories for the input <see cref="Stream"/> instances to read</param>
        /// <param name="count">The number of samples to parse</param>
        private static (float[,], float[,]) ParseSamples((Func<Stream> X, Func<Stream> Y) factory, int count)
        {
            // Input checks
            using (Stream inputs = factory.X(), labels = factory.Y())
            {
                using (GZipStream
                    xGzip = new GZipStream(inputs, CompressionMode.Decompress),
                    yGzip = new GZipStream(labels, CompressionMode.Decompress))
                {
                    float[,]
                        x = new float[count, 784],
                        y = new float[count, 10];
                    xGzip.Read(new byte[16], 0, 16);
                    yGzip.Read(new byte[8], 0, 8);
                    for (int i = 0; i < count; i++)
                    {
                        // Read the image pixel values
                        byte[] temp = new byte[784];
                        xGzip.Read(temp, 0, 784);
                        float[] sample = new float[784];
                        for (int j = 0; j < 784; j++)
                        {
                            sample[j] = temp[j] / 255f;
                        }

                        // Read the label
                        float[,] label = new float[10, 1];
                        int l = yGzip.ReadByte();
                        label[l, 0] = 1;

                        // Copy to result matrices
                        Buffer.BlockCopy(sample, 0, x, sizeof(float) * i * 784, sizeof(float) * 784);
                        Buffer.BlockCopy(label, 0, y, sizeof(float) * i * 10, sizeof(float) * 10);
                    }
                    return (x, y);
                }
            }
        }

        #endregion
    }
}
