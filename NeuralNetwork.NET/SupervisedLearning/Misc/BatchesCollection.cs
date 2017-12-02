using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A class that represents a set of training batches to be used in circular order
    /// </summary>
    internal sealed class BatchesCollection
    {
        // The source list of batches to use
        [NotNull]
        internal readonly TrainingBatch[] Batches;

        /// <summary>
        /// Gets the number of training batches in the current collection
        /// </summary>
        public int Count { get; }

        /// <summary>
        /// Gets the total number of training samples in the current collection
        /// </summary>
        public int Samples { get; }

        // Private constructor from a given collection
        private BatchesCollection([NotNull] TrainingBatch[] batches)
        {
            Batches = batches;
            Count = batches.Length;
            Samples = batches.Sum(b => b.X.GetLength(0));
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection FromDataset((float[,] X ,float[,] Y) dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            int
                samples = dataset.X.GetLength(0),
                w = dataset.X.GetLength(1),
                wy = dataset.Y.GetLength(1);
            if (samples != dataset.Y.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(dataset), "The number of samples must be the same in both x and y");

            // Prepare the different batches
            int
                nBatches = samples / size,
                nBatchMod = samples % size;
            bool oddBatchPresent = nBatchMod > 0;
            TrainingBatch[] batches = new TrainingBatch[nBatches + (oddBatchPresent ? 1 : 0)];
            for (int i = 0; i < batches.Length; i++)
            {
                if (oddBatchPresent && i == batches.Length - 1)
                {
                    float[,]
                        batch = new float[nBatchMod, w],
                        batchY = new float[nBatchMod, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(float) * (dataset.X.Length - batch.Length), batch, 0, sizeof(float) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(float) * (dataset.Y.Length - batchY.Length), batchY, 0, sizeof(float) * batchY.Length);
                    batches[batches.Length - 1] = new TrainingBatch(batch, batchY);
                }
                else
                {
                    float[,]
                        batch = new float[size, w],
                        batchY = new float[size, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(float) * i * batch.Length, batch, 0, sizeof(float) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(float) * i * batchY.Length, batchY, 0, sizeof(float) * batchY.Length);
                    batches[i] = new TrainingBatch(batch, batchY);
                }
            }
            return new BatchesCollection(batches);
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection FromDataset([NotNull] IEnumerable<(float[] X, float[] Y)> dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            TrainingBatch[] batches = dataset.AsParallel().Partition(size).Select(partition =>
            {
                int
                    wx = partition[0].X.Length,
                    wy = partition[0].Y.Length;
                float[,]
                    xBatch = new float[partition.Count, wx],
                    yBatch = new float[partition.Count, wy];
                for (int i = 0; i < partition.Count; i++)
                {
                    Buffer.BlockCopy(partition[i].X, 0, xBatch, sizeof(float) * i * wx, sizeof(float) * wx);
                    Buffer.BlockCopy(partition[i].Y, 0, yBatch, sizeof(float) * i * wy, sizeof(float) * wy);
                }
                return new TrainingBatch(xBatch, yBatch);
            }).ToArray();
            return new BatchesCollection(batches);
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection FromDataset([NotNull] IReadOnlyList<(float[] X, float[] Y)> dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            int
                samples = dataset.Count,
                w = dataset[0].X.Length,
                wy = dataset[0].Y.Length;
            if (dataset.Any(t => t.X.Length != w || t.Y.Length != wy)) throw new ArgumentException("The number of features in each sample must be the same");

            // Prepare the different batches
            int
                nBatches = samples / size,
                nBatchMod = samples % size;
            bool oddBatchPresent = nBatchMod > 0;
            TrainingBatch[] batches = new TrainingBatch[nBatches + (oddBatchPresent ? 1 : 0)];
            for (int i = 0; i < batches.Length; i++)
            {
                if (oddBatchPresent && i == batches.Length - 1)
                {
                    float[,]
                        batch = new float[nBatchMod, w],
                        batchY = new float[nBatchMod, wy];
                    int sampleOffset = i * size;
                    for (int j = 0; j < nBatchMod; j++)
                    {
                        int targetSample = sampleOffset + j;
                        Buffer.BlockCopy(dataset[targetSample].X, 0, batch, sizeof(float) * j * w, sizeof(float) * w);
                        Buffer.BlockCopy(dataset[targetSample].Y, 0, batchY, sizeof(float) * j * wy, sizeof(float) * wy);
                    }
                    batches[batches.Length - 1] = new TrainingBatch(batch, batchY);
                }
                else
                {
                    float[,]
                        batch = new float[size, w],
                        batchY = new float[size, wy];
                    int sampleOffset = i * size;
                    for (int j = 0; j < size; j++)
                    {
                        int targetSample = sampleOffset + j;
                        Buffer.BlockCopy(dataset[targetSample].X, 0, batch, sizeof(float) * j * w, sizeof(float) * w);
                        Buffer.BlockCopy(dataset[targetSample].Y, 0, batchY, sizeof(float) * j * wy, sizeof(float) * wy);
                    }
                    batches[i] = new TrainingBatch(batch, batchY);
                }
            }
            return new BatchesCollection(batches);
        }

        // Cross-shuffles the current dataset
        private void CrossShuffle()
        {
            // Select the couples to cross-shuffle
            int[] indexes = Enumerable.Range(0, Batches.Length).ToArray();
            indexes.Shuffle();
            List<(int, int)> couples = new List<(int, int)>();
            for (int i = 0; i < indexes.Length - 1; i += 2)
            {
                couples.Add((indexes[i], indexes[i + 1]));
            }

            // Cross-shuffle the pairs of lists in parallel
            bool result = Parallel.For(0, couples.Count, i =>
            {
                (int a, int b) = couples[i];
                TrainingBatch setA = Batches[a], setB = Batches[b];
                int
                    hA = setA.X.GetLength(0),
                    wx = setA.X.GetLength(1),
                    wy = setA.Y.GetLength(1),
                    hB = setB.X.GetLength(0),
                    bound = hA > hB ? hB : hA;
                float[]
                    tempX = new float[wx],
                    tempY = new float[wy];
                while (bound > 1)
                {
                    int k = ThreadSafeRandom.NextInt(max: bound);
                    bound--;
                    TrainingBatch
                        targetA = ThreadSafeRandom.NextBool() ? setA : setB,
                        targetB = ThreadSafeRandom.NextBool() ? setA : setB;

                    // Rows from A[k] to temp
                    Buffer.BlockCopy(targetA.X, sizeof(float) * wx * k, tempX, 0, sizeof(float) * wx);
                    Buffer.BlockCopy(targetA.Y, sizeof(float) * wy * k, tempY, 0, sizeof(float) * wy);

                    // Rows from B[bound] to A[k]
                    Buffer.BlockCopy(targetB.X, sizeof(float) * wx * bound, targetA.X, sizeof(float) * wx * k, sizeof(float) * wx);
                    Buffer.BlockCopy(targetB.Y, sizeof(float) * wy * bound, targetA.Y, sizeof(float) * wy * k, sizeof(float) * wy);

                    // Rows from temp to B[bound]
                    Buffer.BlockCopy(tempX, 0, targetB.X, sizeof(float) * wx * bound, sizeof(float) * wx);
                    Buffer.BlockCopy(tempY, 0, targetB.Y, sizeof(float) * wy * bound, sizeof(float) * wy);
                }
            }).IsCompleted;
            if (!result) throw new InvalidOperationException("Failed to perform the parallel loop");

            // Shuffle the main list
            Batches.Shuffle();
        }

        /// <summary>
        /// Shuffles the current dataset and returns a new sequence of batches
        /// </summary>
        /// <returns></returns>
        [Pure, NotNull]
        public IEnumerable<TrainingBatch> NextEpoch()
        {
            CrossShuffle();
            return Batches;
        }
    }
}
