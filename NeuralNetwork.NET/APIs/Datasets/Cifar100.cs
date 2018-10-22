using System;
using System.Collections.Generic;
using System.IO;
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
    /// A static class that provides quick access to the CIFAR-100 database, see <a href="https://www.cs.toronto.edu/~kriz/cifar.html">cs.toronto.edu/~kriz/cifar.html</a>
    /// </summary>
    public static class Cifar100
    {
        #region Constants

        // The number of training samples in the training .bin file
        private const int TrainingSamplesInBinFile = 50000;

        // The number of test samples in the .bin file
        private const int TestSamplesInBinFile = 10000;

        // 32*32 RGB images
        private const int SampleSize = 3072;

        // A single 32*32 image
        private const int ImageSize = 1024;

        private const int CoarseLabels = 20;

        private const int FineLabels = 100;

        private const string DatasetURL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
        
        private const string TrainingBinFilename = "train.bin";
        
        private const string TestBinFilename = "test.bin";

        #endregion

        /// <summary>
        /// Downloads the CIFAR-100 training datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="mode">The desired output mode for the dataset classes</param>
        /// <param name="callback">The optional progress calback</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITrainingDataset> GetTrainingDatasetAsync(int size, Cifar100ClassificationMode mode = Cifar100ClassificationMode.Fine, [CanBeNull] IProgress<HttpProgress> callback = null, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, callback, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])> data = ParseSamples(map[TrainingBinFilename], TrainingSamplesInBinFile, mode);
            return DatasetLoader.Training(data, size);
        }

        /// <summary>
        /// Downloads the CIFAR-100 test datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="mode">The desired output mode for the dataset classes</param>
        /// <param name="callback">The optional progress calback</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITestDataset> GetTestDatasetAsync(
            [CanBeNull] Action<TrainingProgressEventArgs> progress = null, Cifar100ClassificationMode mode = Cifar100ClassificationMode.Fine, 
            [CanBeNull] IProgress<HttpProgress> callback = null, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, callback, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])> data = ParseSamples(map[TestBinFilename], TestSamplesInBinFile, mode);
            return DatasetLoader.Test(data, progress);
        }

        /// <summary>
        /// Downloads and exports the full CIFAR-100 dataset (both training and test samples) to the target directory
        /// </summary>
        /// <param name="directory">The target directory</param>
        /// <param name="token">The cancellation token for the operation</param>
        [PublicAPI]
        public static async Task<bool> ExportDatasetAsync([NotNull] DirectoryInfo directory, CancellationToken token = default)
        {
            IReadOnlyDictionary<string, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, null, token);
            if (map == null) return false;
            if (!directory.Exists) directory.Create();
            ParallelLoopResult result = Parallel.ForEach(new (string Name, int Count)[]
            {
                (TrainingBinFilename, TrainingSamplesInBinFile),
                (TestBinFilename, TestSamplesInBinFile)
            }, (pair, state) =>
            {
                ExportSamples(directory, (pair.Name, map[pair.Name]), pair.Count, token);
                if (token.IsCancellationRequested) state.Stop();
            });
            return result.IsCompleted && !token.IsCancellationRequested;
        }

        #region Tools

        /// <summary>
        /// Parses a CIFAR-100 .bin file
        /// </summary>
        /// <param name="factory">A <see cref="Func{TResult}"/> that returns the <see cref="Stream"/> to read</param>
        /// <param name="count">The number of samples to parse</param>
        /// <param name="mode">The desired output mode for the dataset classes</param>
        private static unsafe IReadOnlyList<(float[], float[])> ParseSamples(Func<Stream> factory, int count, Cifar100ClassificationMode mode)
        {
            // Calculate the output size
            int outputs = (mode.HasFlag(Cifar100ClassificationMode.Coarse) ? CoarseLabels : 0) +
                          (mode.HasFlag(Cifar100ClassificationMode.Fine) ? FineLabels : 0);
            if (outputs == 0) throw new ArgumentOutOfRangeException(nameof(mode), "The input mode isn't valid");
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
                            y = new float[outputs];

                        // Label
                        int
                            coarse = stream.ReadByte(),
                            fine = stream.ReadByte();
                        switch (mode)
                        {
                            case Cifar100ClassificationMode.Coarse:
                                y[coarse] = 1;
                                break;
                            case Cifar100ClassificationMode.Fine:
                                y[fine] = 1;
                                break;
                            default:
                                y[coarse] = 1;
                                y[CoarseLabels + fine] = 1;
                                break;
                        }

                        // Image data
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
        /// Exports a CIFAR-100 .bin file
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
                        int
                            coarse = stream.ReadByte(),
                            fine = stream.ReadByte();
                        stream.Read(temp, 0, SampleSize);
                        using (Image<Rgb24> image = new Image<Rgb24>(32, 32))
                            fixed (Rgb24* p0 = image.GetPixelSpan())
                            {
                                for (int j = 0; j < ImageSize; j++)
                                    p0[j] = new Rgb24(ptemp[j], ptemp[j + ImageSize], ptemp[j + 2 * ImageSize]);
                                using (FileStream file = File.OpenWrite(Path.Combine(folder.FullName, $"[{source.Name}][{i}][{coarse}][{fine}].bmp")))
                                    image.SaveAsBmp(file);
                            }
                    }
                }
            }
        }

        #endregion

        /// <summary>
        /// An <see langword="enum"/> indicating the type of outputs for a CIFAR-100 dataset instance
        /// </summary>
        [Flags]
        public enum Cifar100ClassificationMode
        {
            /// <summary>
            /// Each sample is classified using the 20 available superclasses
            /// </summary>
            Coarse = 0b01,

            /// <summary>
            /// Each sample is classified using the 100 available narrow classes
            /// </summary>
            Fine = 0b10
        }
    }
}
