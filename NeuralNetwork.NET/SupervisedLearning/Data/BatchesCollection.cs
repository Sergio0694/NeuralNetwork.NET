using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.SupervisedLearning.Data
{
    /// <summary>
    /// A class that represents a set of samples batches to be used in circular order
    /// </summary>
    internal sealed class BatchesCollection : IDataset
    {
        /// <summary>
        /// Gets the collection of samples batches to use
        /// </summary>
        [NotNull]
        public SamplesBatch[] Batches { get; }

        #region Interface

        /// <inheritdoc/>
        public int Count { get; }

        /// <inheritdoc/>
        public (Span<float> X, Span<float> Y) this[int i]
        {
            get
            {
                if (i < 0 || i > Count - 1) throw new ArgumentOutOfRangeException(nameof(i), "The target index is not valid");
                ref readonly SamplesBatch batch = ref Batches[i / Batches.Length];
                return (batch.X.Slice(i), batch.Y.Slice(i));
            }
        }

        /// <inheritdoc/>
        public IEnumerator<(Span<float> X, Span<float> Y)> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
                yield return this[i];
        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <inheritdoc/>
        public long ByteSize => sizeof(float) * Count * (this[0].X.Length + this[0].Y.Length);

        #endregion

        #region Initialization

        // Private constructor from a given collection
        private BatchesCollection([NotNull] SamplesBatch[] batches)
        {
            Batches = batches;
            Count = batches.Sum(b => b.X.GetLength(0));
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection From((float[,] X, float[,] Y) dataset, int size)
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
            SamplesBatch[] batches = new SamplesBatch[nBatches + (oddBatchPresent ? 1 : 0)];
            for (int i = 0; i < batches.Length; i++)
            {
                if (oddBatchPresent && i == batches.Length - 1)
                {
                    float[,]
                        batch = new float[nBatchMod, w],
                        batchY = new float[nBatchMod, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(float) * (dataset.X.Length - batch.Length), batch, 0, sizeof(float) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(float) * (dataset.Y.Length - batchY.Length), batchY, 0, sizeof(float) * batchY.Length);
                    batches[batches.Length - 1] = new SamplesBatch(batch, batchY);
                }
                else
                {
                    float[,]
                        batch = new float[size, w],
                        batchY = new float[size, wy];
                    Buffer.BlockCopy(dataset.X, sizeof(float) * i * batch.Length, batch, 0, sizeof(float) * batch.Length);
                    Buffer.BlockCopy(dataset.Y, sizeof(float) * i * batchY.Length, batchY, 0, sizeof(float) * batchY.Length);
                    batches[i] = new SamplesBatch(batch, batchY);
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
        public static BatchesCollection From([NotNull] IEnumerable<Func<(float[] X, float[] Y)>> dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            return new BatchesCollection(dataset.AsParallel().Select(f => f()).Partition(size).Select(SamplesBatch.From).ToArray());
        }

        /// <summary>
        /// Creates a series of batches from the input dataset and expected results
        /// </summary>
        /// <param name="dataset">The source dataset to create the batches</param>
        /// <param name="size">The desired batch size</param>
        /// <exception cref="ArgumentOutOfRangeException">The dataset and result matrices have a different number of rows</exception>
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static BatchesCollection From([NotNull] IEnumerable<(float[] X, float[] Y)> dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            return new BatchesCollection(dataset.ToArray().AsParallel().Partition(size).Select(SamplesBatch.From).ToArray());
        }

        #endregion

        #region Shuffling

        /// <summary>
        /// Cross-shuffles the current dataset (shuffles samples in each batch, then shuffles the batches list)
        /// </summary>
        public unsafe void CrossShuffle()
        {
            // Select the couples to cross-shuffle
            int* indexes = stackalloc int[Batches.Length];
            for (int i = 0; i < Batches.Length; i++) indexes[i] = i;
            int n = Batches.Length;
            while (n > 1)
            {
                int k = ThreadSafeRandom.NextInt(max: n);
                n--;
                int value = indexes[k];
                indexes[k] = indexes[n];
                indexes[n] = value;
            }

            // Cross-shuffle the pairs of lists in parallel
            void Kernel(int i)
            {
                int a = indexes[i * 2], b = indexes[i * 2 + 1];
                SamplesBatch setA = Batches[a], setB = Batches[b];
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
                    SamplesBatch
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
            }
            Parallel.For(0, Batches.Length / 2, Kernel).AssertCompleted();

            // Shuffle the main list
            n = Batches.Length;
            while (n > 1)
            {
                int k = ThreadSafeRandom.NextInt(max: n);
                n--;
                SamplesBatch value = Batches[k];
                Batches[k] = Batches[n];
                Batches[n] = value;
            }
        }

        #endregion
    }
}
