using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace MnistDatasetToolkit
{
    /// <summary>
    /// A simple static class that downloads and parses the MNIST datasets
    /// </summary>
    public static class DataParser
    {
        #region Constants

        private const String MnistHttpRootPath = "http://yann.lecun.com/exdb/mnist/";

        private const String TrainingSetValuesFilename = "train-images-idx3-ubyte.gz";

        private const String TrainingSetLabelsFilename = "train-labels-idx1-ubyte.gz";

        private const String TestSetValuesFilename = "t10k-images-idx3-ubyte.gz";

        private const String TestSetLabelsFilename = "t10k-labels-idx1-ubyte.gz";

        #endregion

        #region Tools

        /// <summary>
        /// Extracts the name of the saved file from the given original filename
        /// </summary>
        /// <param name="filename">The MNIST filename to convert</param>
        private static String GetDecompressedDatasetFilename([NotNull] String filename) => filename.Split('.')[0].Replace("-idx", ".idx");

        /// <summary>
        /// Downloads the MNIST database and saves it in the current folder
        /// </summary>
        private static String TryDownloadDataset()
        {
            String
                code = Assembly.GetExecutingAssembly().Location,
                dll = Path.GetFullPath(code),
                root = Path.GetDirectoryName(dll),
                path = Path.Combine(root, "MNIST");
            IEnumerable<String> folders = Directory.EnumerateDirectories(root);
            if (!folders.Any(folder => Path.GetFileName(folder).Equals("MNIST")))
            {
                Directory.CreateDirectory(path);
                Parallel.ForEach(new[] { TrainingSetValuesFilename, TrainingSetLabelsFilename, TestSetValuesFilename, TestSetLabelsFilename }, (name, state) =>
                {
                    using (HttpClient client = new HttpClient())
                    using (Stream raw = client.GetStreamAsync($"{MnistHttpRootPath}{name}").Result)
                    using (FileStream file = File.Create(Path.Combine(path, GetDecompressedDatasetFilename(name))))
                    {
                        byte[] block = new byte[1024];
                        int read;
                        while ((read = raw.Read(block, 0, 1024)) > 0)
                        {
                            file.Write(block, 0, read);
                        }
                    }
                });
            }
            return path;
        }

        #endregion

        /// <summary>
        /// Loads the MNIST dataset and returns both the training dataset and the test dataset
        /// </summary>
        [PublicAPI]
        [MustUseReturnValue]
        public static ((double[,] X, double[,] Y) TrainingData, (double[,] X, double[,] Y) TestData) LoadDatasets()
        {
            String path = TryDownloadDataset();
            (double[,], double[,]) ParseSamples(String valuePath, String labelsPath, int count)
            {
                double[,]
                    x = new double[count, 784],
                    y = new double[count, 10];
                using (FileStream
                    xStream = File.OpenRead(Path.Combine(path, GetDecompressedDatasetFilename(valuePath))),
                    yStream = File.OpenRead(Path.Combine(path, GetDecompressedDatasetFilename(labelsPath))))
                using (GZipStream
                    xGzip = new GZipStream(xStream, CompressionMode.Decompress),
                    yGzip = new GZipStream(yStream, CompressionMode.Decompress))
                {
                    xGzip.Read(new byte[16], 0, 16);
                    yGzip.Read(new byte[8], 0, 8);
                    for (int i = 0; i < count; i++)
                    {
                        // Read the image pixel values
                        byte[] temp = new byte[784];
                        xGzip.Read(temp, 0, 784);
                        double[] sample = new double[784];
                        for (int j = 0; j < 784; j++)
                        {
                            sample[j] = temp[j] / 255d;
                        }

                        // Read the label
                        double[,] label = new double[10, 1];
                        int l = yGzip.ReadByte();
                        label[l, 0] = 1;

                        // Copy to result matrices
                        Buffer.BlockCopy(sample, 0, x, sizeof(double) * i * 784, sizeof(double) * 784);
                        Buffer.BlockCopy(label, 0, y, sizeof(double) * i * 10, sizeof(double) * 10);
                    }
                    return (x, y);
                }
            }
            return (ParseSamples(TrainingSetValuesFilename, TrainingSetLabelsFilename, 60_000),
                    ParseSamples(TestSetValuesFilename, TestSetLabelsFilename, 10_000));
        }
    }
}
