using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace MnistDatasetToolkit
{
    public static class DataParser
    {
        private const int SamplesPixelSize = 784;

        public static unsafe IReadOnlyList<double[,]> ConvertDatasetTo2dImages([NotNull] double[,] dataset)
        {
            int samples = dataset.GetLength(0);
            double[][,] raw = new double[samples][,];
            Parallel.For(0, raw.Length, i =>
            {
                double[,] _2d = new double[28, 28];
                fixed (double* p = _2d, pr = dataset)
                {
                    int start = i * 784;
                    for (int j = 0; j < 28; j++)
                        for (int k = 0; k < 28; k++)
                        {
                            int offset = j * 28 + k;
                            p[offset] = pr[start + offset];
                        }
                }
                raw[i] = _2d;
            });
            return raw;
        }

        public static (double[,], double[,]) ParseDataset(
            [NotNull] String datasetValuesPath,
            [NotNull] String datasetLabelsPath,
            int? limit = null) => (ParseDataset(datasetValuesPath, limit), ParseY(datasetLabelsPath, limit));

        [NotNull]
        private static unsafe double[,] ParseDataset([NotNull] String path, int? limit = null)
        {
            using (FileStream data = File.OpenRead(path))
            {
                // Seek to the size info (0x4)
                int size;
                if (limit == null)
                {
                    data.Seek(4, SeekOrigin.Begin);
                    byte[] length = new byte[4];
                    data.Read(length, 0, 4);
                    size = length.ToLittleEndian();
                    data.Seek(8, SeekOrigin.Current);
                }
                else
                {
                    data.Seek(16, SeekOrigin.Begin);
                    size = limit.Value;
                }
                double[,] dataset = new double[size, SamplesPixelSize]; // n 28*28 images

                // Parse the sample images
                for (int i = 0; i < size; i++)
                {
                    // Read the image pixel values
                    byte[] temp = new byte[SamplesPixelSize];
                    data.Read(temp, 0, SamplesPixelSize);
                    fixed (double* p = dataset)
                    fixed (byte* t = temp)
                    {
                        // Copy the values in the output matrix
                        int offset = i * SamplesPixelSize;
                        for (int j = 0; j < SamplesPixelSize; j++) p[offset + j] = t[j] / 255d; // Normalize to [0..1]
                    }
                }
                return dataset;
            }
        }

        private static double[,] ParseY([NotNull] String path, int? limit = null)
        {
            using (FileStream data = File.OpenRead(path))
            {
                int size;
                if (limit == null)
                {
                    data.Seek(4, SeekOrigin.Begin);
                    byte[] length = new byte[4];
                    data.Read(length, 0, 4);
                    size = length.ToLittleEndian();
                }
                else
                {
                    data.Seek(8, SeekOrigin.Begin);
                    size = limit.Value;
                }
                double[,] dataset = new double[size, 10]; // n samples
                for (int i = 0; i < size; i++)
                {
                    int value = data.ReadByte();
                    dataset[i, value] = 1;
                }
                return dataset;
            }
        }

        // Converts a big endian byte array into a little endian integer
        private static int ToLittleEndian([NotNull] this byte[] bytes) => bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
    }
}
