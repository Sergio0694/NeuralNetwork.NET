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

namespace NeuralNetworkNET.APIs.Datasets
{
    /// <summary>
    /// A static class that provides quick access to the CIFAR-100 database, see <a href="https://www.cs.toronto.edu/~kriz/cifar.html">cs.toronto.edu/~kriz/cifar.html</a>
    /// </summary>
    public static class Cifar100
    {
        #region Constants

        // The number of training samples in each extracted .bin file
        private const int TrainingSamplesInBinFiles = 10000;

        // 32*32 RGB images
        private const int SampleSize = 3072;

        private const int CoarseLabels = 20;

        private const int FineLabels = 100;

        private const String DatasetURL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";

        [NotNull, ItemNotNull]
        private static readonly IReadOnlyList<String> TrainingBinFilenames = Enumerable.Range(1, 5).Select(i => $"data_batch_{i}.bin").ToArray();

        private const String TestBinFilename = "test_batch.bin";

        #endregion

        /// <summary>
        /// Downloads the CIFAR-100 training datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="mode">The desired output mode for the dataset classes</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITrainingDataset> GetTrainingDatasetAsync(int size, Cifar100ClassificationMode mode = Cifar100ClassificationMode.Fine, CancellationToken token = default)
        {
            IReadOnlyDictionary<String, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])>[] data = new IReadOnlyList<(float[], float[])>[TrainingBinFilenames.Count];
            Parallel.For(0, TrainingBinFilenames.Count, i => data[i] = ParseSamples(map[TrainingBinFilenames[i]], TrainingSamplesInBinFiles, mode)).AssertCompleted();
            return DatasetLoader.Training(data.Skip(1).Aggregate(data[0] as IEnumerable<(float[], float[])>, (s, l) => s.Concat(l)), size);
        }

        /// <summary>
        /// Downloads the CIFAR-100 test datasets and returns a new <see cref="ITestDataset"/> instance
        /// </summary>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="mode">The desired output mode for the dataset classes</param>
        /// <param name="token">An optional cancellation token for the operation</param>
        [PublicAPI]
        [Pure, ItemCanBeNull]
        public static async Task<ITestDataset> GetTestDatasetAsync([CanBeNull] Action<TrainingProgressEventArgs> progress = null, Cifar100ClassificationMode mode = Cifar100ClassificationMode.Fine, CancellationToken token = default)
        {
            IReadOnlyDictionary<String, Func<Stream>> map = await DatasetsDownloader.GetArchiveAsync(DatasetURL, token);
            if (map == null) return null;
            IReadOnlyList<(float[], float[])> data = ParseSamples(map[TestBinFilename], TrainingSamplesInBinFiles, mode);
            return DatasetLoader.Test(data, progress);
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
                            for (int j = 0; j < SampleSize; j++)
                                px[j] = ptemp[j] / 255f; // Normalized samples
                        }
                        data[i] = (x, y);
                    }
                }
                return data;
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
