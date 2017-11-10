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
        private readonly SupervisedDataset[] Batches;

        /// <summary>
        /// Gets the number of training batches in the current collection
        /// </summary>
        public int Count { get; }

        /// <summary>
        /// Gets the total number of training samples in the current collection
        /// </summary>
        public int Samples { get; }

        // Private random instance to shuffle the batches
        private Random RandomProvider { get; }

        // Private constructor from a given collection
        private BatchesCollection([NotNull] SupervisedDataset[] batches)
        {
            Batches = batches;
            Count = batches.Length;
            Samples = batches.Sum(b => b.X.GetLength(0));
            RandomProvider = new Random(batches.Aggregate(batches.GetHashCode(), (s, h) => s ^ h.GetHashCode()));
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection FromDataset(SupervisedDataset dataset, int size)
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
            SupervisedDataset[] batches = new SupervisedDataset[nBatches + (oddBatchPresent ? 1 : 0)];
            for (int i = 0; i < batches.Length; i++)
            {
                if (oddBatchPresent && i == batches.Length - 1)
                {
                    double[,]
                        batch = new double[nBatchMod, w],
                        batchY = new double[nBatchMod, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(double) * (dataset.X.Length - batch.Length), batch, 0, sizeof(double) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(double) * (dataset.Y.Length - batchY.Length), batchY, 0, sizeof(double) * batchY.Length);
                    batches[batches.Length - 1] = new SupervisedDataset(batch, batchY);
                }
                else
                {
                    double[,]
                        batch = new double[size, w],
                        batchY = new double[size, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(double) * i * batch.Length, batch, 0, sizeof(double) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(double) * i * batchY.Length, batchY, 0, sizeof(double) * batchY.Length);
                    batches[i] = new SupervisedDataset(batch, batchY);
                }
            }
            return new BatchesCollection(batches);
        }

        // Cross-shuffles the current dataset
        private void CrossShuffle()
        {
            // Select the couples to cross-shuffle
            int[] indexes = Enumerable.Range(0, Batches.Length).ToArray();
            indexes.Shuffle(RandomProvider);
            List<(int, int)> couples = new List<(int, int)>();
            for (int i = 0; i < indexes.Length - 1; i += 2)
            {
                couples.Add((indexes[i], indexes[i + 1]));
            }

            // Cross-shuffle the pairs of lists in parallel
            bool result = Parallel.For(0, couples.Count, i =>
            {
                (int a, int b) = couples[i];
                SupervisedDataset setA = Batches[a], setB = Batches[b];
                Random r = new Random(a ^ b ^ setA.GetHashCode() ^ setB.GetHashCode());
                int
                    hA = setA.X.GetLength(0),
                    wx = setA.X.GetLength(1),
                    wy = setA.Y.GetLength(1),
                    hB = setB.X.GetLength(0),
                    bound = hA > hB ? hB : hA;
                double[]
                    tempX = new double[wx],
                    tempY = new double[wy];
                while (bound > 1)
                {
                    int k = r.Next(0, bound) % bound;
                    bound--;
                    SupervisedDataset
                        targetA = r.NextBool() ? setA : setB,
                        targetB = r.NextBool() ? setA : setB;

                    // Rows from A[k] to temp
                    Buffer.BlockCopy(targetA.X, sizeof(double) * wx * k, tempX, 0, sizeof(double) * wx);
                    Buffer.BlockCopy(targetA.Y, sizeof(double) * wy * k, tempY, 0, sizeof(double) * wy);

                    // Rows from B[bound] to A[k]
                    Buffer.BlockCopy(targetB.X, sizeof(double) * wx * bound, targetA.X, sizeof(double) * wx * k, sizeof(double) * wx);
                    Buffer.BlockCopy(targetB.Y, sizeof(double) * wy * bound, targetA.Y, sizeof(double) * wy * k, sizeof(double) * wy);

                    // Rows from temp to B[bound]
                    Buffer.BlockCopy(tempX, 0, targetB.X, sizeof(double) * wx * bound, sizeof(double) * wx);
                    Buffer.BlockCopy(tempY, 0, targetB.Y, sizeof(double) * wy * bound, sizeof(double) * wy);
                }
            }).IsCompleted;
            if (!result) throw new InvalidOperationException("Failed to perform the parallel loop");

            // Shuffle the main list
            Batches.Shuffle(RandomProvider);
        }

        /// <summary>
        /// Shuffles the current dataset and returns a new sequence of batches
        /// </summary>
        /// <returns></returns>
        [Pure, NotNull]
        public IEnumerable<SupervisedDataset> NextEpoch()
        {
            CrossShuffle();
            return Batches;
        }
    }
}
