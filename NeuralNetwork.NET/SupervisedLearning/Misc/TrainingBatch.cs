using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.SupervisedLearning.Misc
{
    /// <summary>
    /// A simple struct that contains a reference to the data in a training batch
    /// </summary>
    internal struct TrainingBatch
    {
        /// <summary>
        /// Gets the training data for the current batch
        /// </summary>
        [NotNull]
        public double[,] X { get; }

        /// <summary>
        /// Gets the expected results for the current batch
        /// </summary>
        [NotNull]
        public double[,] Y { get; }

        /// <summary>
        /// Creates a new training batch with the given data
        /// </summary>
        /// <param name="x">The batch data</param>
        /// <param name="y">The batch expected results</param>
        private TrainingBatch([NotNull] double[,] x, [NotNull] double[,] y)
        {
            if (x.GetLength(0) != y.GetLength(0)) throw new ArgumentException("The number of samples in the batch data and results must be the same");
            X = x;
            Y = y;
        }

        /// <summary>
        /// A class that represents a set of training batches to be used in circular order
        /// </summary>
        public sealed class BatchesCollection
        {
            // The source list of batches to use
            [NotNull]
            private readonly IReadOnlyList<TrainingBatch> Batches;

            // Private constructor from a given collection
            private BatchesCollection([NotNull] IReadOnlyList<TrainingBatch> batches) => Batches = batches;

            // Index to select the next item
            private int _Index;

            /// <summary>
            /// Selects a new <see cref="TrainingBatch"/> from the current dataset
            /// </summary>
            [MustUseReturnValue]
            public TrainingBatch Next()
            {
                TrainingBatch pick = Batches[_Index++];
                _Index %= Batches.Count;
                return pick;
            }

            /// <summary>
            /// Creates a series of batches from the input dataset and expected results
            /// </summary>
            /// <param name="x">The original dataset</param>
            /// <param name="y">The expected results for the input dataset</param>
            /// <param name="size">The desired batch size</param>
            /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
            [NotNull]
            [CollectionAccess(CollectionAccessType.Read)]
            public static BatchesCollection FromDataset([NotNull] double[,] x, [NotNull] double[,] y, int size)
            {
                // Local parameters
                int
                    samples = x.GetLength(0),
                    w = x.GetLength(1),
                    wy = y.GetLength(1);
                if (samples != y.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(y), "The number of samples must be the same in both x and y");
                if (size == samples) return new BatchesCollection(new[] { new TrainingBatch(x, y) }); // Fake batch with the whole dataset

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
                        double[,]
                            batch = new double[nBatchMod, w],
                            batchY = new double[nBatchMod, wy];
                        Buffer.BlockCopy(x, sizeof(double) * (x.Length - batch.Length), batch, 0, sizeof(double) * batch.Length);
                        Buffer.BlockCopy(y, sizeof(double) * (y.Length - batchY.Length), batchY, 0, sizeof(double) * batchY.Length);
                        batches[batches.Length - 1] = new TrainingBatch(batch, batchY);
                    }
                    else
                    {
                        double[,]
                            batch = new double[size, w],
                            batchY = new double[size, wy];
                        Buffer.BlockCopy(x, sizeof(double) * i * batch.Length, batch, 0, sizeof(double) * batch.Length);
                        Buffer.BlockCopy(y, sizeof(double) * i * batchY.Length, batchY, 0, sizeof(double) * batchY.Length);
                        batches[i] = new TrainingBatch(batch, batchY);
                    }
                }
                return new BatchesCollection(batches);
            }
        }
    }
}