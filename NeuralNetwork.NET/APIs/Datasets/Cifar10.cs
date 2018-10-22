using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworkNET.APIs.Datasets
{
    /// <summary>
    /// A static class that provides quick access to the CIFAR-10 database, see <a href="https://www.cs.toronto.edu/~kriz/cifar.html">cs.toronto.edu/~kriz/cifar.html</a>
    /// </summary>
    public static class Cifar10
    {
        #region Constants

        // The number of training samples in each extracted .bin file
        private const int TrainingSamplesInBinFiles = 10000;

        // 32*32 RGB images
        private const int SampleSize = 3072;

        // A single 32*32 image
        private const int ImageSize = 1024;

        private const string DatasetURL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

        [NotNull, ItemNotNull]
        private static readonly IReadOnlyList<string> TrainingBinFilenames = Enumerable.Range(1, 5).Select(i => $"data_batch_{i}.bin").ToArray();
        
        private const string TestBinFilename = "test_batch.bin";

        #endregion

        /// <summary>
        /// Downloads the CIFAR-10 training datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="callback">The optional progress calback</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITrainingDataset> GetTrainingDatasetAsync(int size, [CanBeNull] IProgress<HttpProgress> callback = null, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, callback, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])>[] data = new IReadOnlyList<(float[], float[])>[TrainingBinFilenames.Count];
            Parallel.For(0, TrainingBinFilenames.Count, i => data[i] = ParseSamples(map[TrainingBinFilenames[i]], TrainingSamplesInBinFiles)).AssertCompleted();
            return DatasetLoader.Training(data.Skip(1).Aggregate(data[0] as IEnumerable<(float[], float[])>, (s, l) => s.Concat(l)), size);
        }

        /// <summary>
        /// Downloads the CIFAR-10 test datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="callback">The optional progress calback</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITestDataset> GetTestDatasetAsync([CanBeNull] Action<TrainingProgressEventArgs> progress = null, [CanBeNull] IProgress<HttpProgress> callback = null, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, callback, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])> data = ParseSamples(map[TestBinFilename], TrainingSamplesInBinFiles);
            return DatasetLoader.Test(data, progress);
        }

        /// <summary>
        /// Downloads and exports the full CIFAR-10 dataset (both training and test samples) to the target directory
        /// </summary>
        /// <param name="directory">The target directory</param>
        /// <param name="token">The cancellation token for the operation</param>
        [PublicAPI]
        public static async Task<bool> ExportDatasetAsync([NotNull] DirectoryInfo directory, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, null, token);
            if (map == null) return false;
            if (!directory.Exists) directory.Create();
            ParallelLoopResult result = Parallel.ForEach(TrainingBinFilenames.Concat(new[] { TestBinFilename }), (name, state) =>
            {
                ExportSamples(directory, (name, map[name]), TrainingSamplesInBinFiles, token);
                if (token.IsCancellationRequested) state.Stop();
            });
            return result.IsCompleted && !token.IsCancellationRequested;
        }

        #region Tools

        /// <summary>
        /// Parses a CIFAR-10 .bin file
        /// </summary>
        /// <param name="factory">A <see cref="Func{TResult}"/> that returns the <see cref="Stream"/> to read</param>
        /// <param name="count">The number of samples to parse</param>
        private static unsafe IReadOnlyList<(float[], float[])> ParseSamples([NotNull] Func<Stream> factory, int count)
        {
            using (Stream stream = factory())
            {
                (float[], float[])[] data = new (float[], float[])[count];
                byte[] temp = new byte[SampleSize];
                fixed (byte* ptemp = temp)
                {
                    for (int i = 0; i < count; i++)
                    {
                        float[]
                            x = new float[SampleSize],
                            y = new float[10];
                        y[stream.ReadByte()] = 1; // Label
                        fixed (float* px = x)
                        {
                            stream.Read(temp, 0, SampleSize);
                            for (int j = 0; j < ImageSize; j++)
                            {
                                px[j] = ptemp[j] / 255f; // Normalized samples
                                px[j] = ptemp[j + ImageSize] / 255f;
                                px[j] = ptemp[j + 2 * ImageSize] / 255f;
                            }
                        }
                        data[i] = (x, y);
                    }
                }
                return data;
            }
        }

        /// <summary>
        /// Exports a CIFAR-10 .bin file
        /// </summary>
        /// <param name="folder">The target folder to use to save the images</param>
        /// <param name="source">The source filename and a <see cref="Func{TResult}"/> that returns the <see cref="Stream"/> to read</param>
        /// <param name="count">The number of samples to parse</param>
        /// <param name="token">A token for the operation</param>
        private static unsafe void ExportSamples([NotNull] DirectoryInfo folder, (string Name, Func<Stream> Factory) source, int count, CancellationToken token)
        {
            using (Stream stream = source.Factory())
            {
                byte[] temp = new byte[SampleSize];
                fixed (byte* ptemp = temp)
                {
                    for (int i = 0; i < count; i++)
                    {
                        if (token.IsCancellationRequested) return;
                        int label = stream.ReadByte();
                        stream.Read(temp, 0, SampleSize);
                        using (Image<Rgb24> image = new Image<Rgb24>(32, 32))
                        fixed (Rgb24* p0 = image.GetPixelSpan())
                        {
                            for (int j = 0; j < ImageSize; j++)
                                p0[j] = new Rgb24(ptemp[j], ptemp[j + ImageSize], ptemp[j + 2 * ImageSize]);
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
