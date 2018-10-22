using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.SupervisedLearning.Data
{
    /// <summary>
    /// A class that represents a set of samples batches to be used in circular order
    /// </summary>
    internal sealed class BatchesCollection : ITrainingDataset
    {
        /// <summary>
        /// Gets the collection of samples batches to use
        /// </summary>
        [NotNull]
        public SamplesBatch[] Batches { get; private set; }

        #region Interface

        /// <inheritdoc/>
        public int Count { get; }

        /// <inheritdoc/>
        public int BatchesCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Batches.Length;
        }

        /// <inheritdoc/>
        public int InputFeatures => Batches[0].X.GetLength(1);

        /// <inheritdoc/>
        public int OutputFeatures => Batches[0].Y.GetLength(1);

        /// <inheritdoc/>
        public DatasetSample this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (i < 0 || i > Count - 1) throw new ArgumentOutOfRangeException(nameof(i), "The target index is not valid");
                ref readonly SamplesBatch batch = ref Batches[i / BatchSize];
                int row = i % BatchSize;
                return new DatasetSample(batch.X.Slice(row), batch.Y.Slice(row));
            }
        }

        /// <inheritdoc/>
        public int BatchSize
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Batches[0].X.GetLength(0);
            set => Reshape(value >= 10 ? value : throw new ArgumentOutOfRangeException(nameof(BatchSize), "The batch size must be greater than or equal to 10"));
        }

        /// <inheritdoc/>
        public long ByteSize => sizeof(float) * Count * (InputFeatures + OutputFeatures);

        /// <inheritdoc/>
        public unsafe int Id
        {
            get
            {
                int[] temp = new int[Count];
                int
                    wx = InputFeatures,
                    wy = OutputFeatures;
                Parallel.For(0, Count, i =>
                {
                    ref readonly SamplesBatch batch = ref Batches[i / BatchSize];
                    int offset = i % BatchSize;
                    fixed (float* px = batch.X, py = batch.Y)
                        temp[i] = new Span<float>(px + offset * wx, wx).GetContentHashCode() ^ 
                                  new Span<float>(py + offset * wy, wy).GetContentHashCode();
                }).AssertCompleted();
                Array.Sort(temp);
                return temp.AsSpan().GetContentHashCode();
            }
        }

        #endregion

        #region Dataset management

        /// <inheritdoc/>
        public void Expand(params Func<float[], float[]>[] factories)
        {
            if (factories.Length < 1) throw new ArgumentException("There haas to be at least one input factory", nameof(factories));
            Batches = From(Batches.SelectMany(b =>
            {
                IEnumerable<Func<(float[], float[])>> Expander()
                {
                    int n = b.X.GetLength(0);
                    for (int i = 0; i < n; i++)
                    {
                        float[]
                            x = b.X.Slice(i),
                            y = b.Y.Slice(i);
                        yield return () => (x, y);
                        foreach (Func<float[], float[]> f in factories)
                            yield return () => (f(x), y);
                    }
                }
                return Expander();
            }), BatchSize).Batches;
        }

        /// <inheritdoc/>
        public (ITrainingDataset, ITestDataset) PartitionWithTest(float ratio, Action<TrainingProgressEventArgs> progress = null)
        {
            int left = CalculatePartitionSize(ratio);
            return (From(Take(0, left), BatchSize), DatasetLoader.Test(Take(left, Count), progress));
        }

        /// <inheritdoc/>
        public (ITrainingDataset, IValidationDataset) PartitionWithValidation(float ratio, float tolerance = 1e-2f, int epochs = 5)
        {
            int left = CalculatePartitionSize(ratio);
            return (From(Take(0, left), BatchSize), DatasetLoader.Validation(Take(left, Count), tolerance, epochs));
        }

        /// <inheritdoc/>
        public ITestDataset ExtractTest(float ratio, Action<TrainingProgressEventArgs> progress = null)
        {
            int left = CalculatePartitionSize(ratio);
            ITestDataset test = DatasetLoader.Test(Take(left, Count), progress);
            Batches = From(Take(0, left), BatchSize).Batches;
            return test;
        }

        /// <inheritdoc/>
        public IValidationDataset ExtractValidation(float ratio, float tolerance = 1e-2f, int epochs = 5)
        {
            int left = CalculatePartitionSize(ratio);
            IValidationDataset validation = DatasetLoader.Validation(Take(left, Count), tolerance, epochs);
            Batches = From(Take(0, left), BatchSize).Batches;
            return validation;
        }

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
        public static unsafe BatchesCollection From((float[,] X, float[,] Y) dataset, int size)
        {
            // Local parameters
            if (size < 10) throw new ArgumentOutOfRangeException(nameof(size), "The batch size can't be smaller than 10");
            int
                samples = dataset.X.GetLength(0),
                wx = dataset.X.GetLength(1),
                wy = dataset.Y.GetLength(1);
            if (samples != dataset.Y.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(dataset), "The number of samples must be the same in both x and y");

            // Prepare the different batches
            int
                nBatches = samples / size,
                nBatchMod = samples % size;
            bool oddBatchPresent = nBatchMod > 0;
            SamplesBatch[] batches = new SamplesBatch[nBatches + (oddBatchPresent ? 1 : 0)];
            fixed (float* px = dataset.X, py = dataset.Y)
            {
                for (int i = 0; i < batches.Length; i++)
                {
                    if (oddBatchPresent && i == batches.Length - 1)
                    {
                        batches[i] = SamplesBatch.From(
                            new Span<float>(px + i * size * wx, nBatchMod * wx),
                            new Span<float>(py + i * size * wy, nBatchMod * wy),
                            wx, wy);
                    }
                    else
                    {
                        batches[i] = SamplesBatch.From(
                            new Span<float>(px + i * size * wx, size * wx),
                            new Span<float>(py + i * size * wy, size * wy),
                            wx, wy);
                    }
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

        #region Misc

        // Reshapes the current dataset with the given batch size
        private unsafe void Reshape(int size)
        {
            // Pin the dataset
            GCHandle*
                xhandles = stackalloc GCHandle[Batches.Length],
                yhandles = stackalloc GCHandle[Batches.Length];
            for (int i = 0; i < Batches.Length; i++)
            {
                xhandles[i] = GCHandle.Alloc(Batches[i].X, GCHandleType.Pinned);
                yhandles[i] = GCHandle.Alloc(Batches[i].Y, GCHandleType.Pinned);
            }

            // Re-partition the current samples
            IEnumerable<SamplesBatch> query =
                from seq in Batches.AsParallel().SelectMany(batch =>
                    from i in Enumerable.Range(0, batch.X.GetLength(0))
                    select (Pin<float>.From(ref batch.X[i, 0]), Pin<float>.From(ref batch.Y[i, 0]))).Partition(size)
                select SamplesBatch.From(seq, InputFeatures, OutputFeatures);
            Batches = query.ToArray();

            // Cleanup
            for (int i = 0; i < Batches.Length; i++)
            {
                xhandles[i].Free();
                yhandles[i].Free();
            }
        }

        // Takes a range of samples from the current dataset
        [Pure, NotNull]
        private IEnumerable<(float[], float[])> Take(int start, int end)
        {
            if (start < 0 || start == end) throw new ArgumentOutOfRangeException(nameof(start));
            if (end > Count) throw new ArgumentOutOfRangeException(nameof(end));
            for (int i = start; i < end; i++)
            {
                DatasetSample sample = this[i];
                yield return (sample.X.ToArray(), sample.Y.ToArray());
            }
        }

        // Computes the size of the first dataset partition given a partition ratio
        [Pure]
        private int CalculatePartitionSize(float ratio)
        {
            if (ratio <= 0 || ratio >= 1) throw new ArgumentOutOfRangeException(nameof(ratio), "The ratio must be in the (0,1) range");
            int left = ((int)(Count * (1 - ratio))).Max(10); // Ensure there are at least 10 elements
            if (Count - left < 10) throw new ArgumentOutOfRangeException(nameof(ratio), "Each partition must have at least 10 samples");
            return left;
        }

        #endregion

        #region Shuffling

        /// <summary>
        /// Cross-shuffles the current dataset (shuffles samples in each batch, then shuffles the batches list)
        /// </summary>
        public unsafe void CrossShuffle()
        {
            // Select the couples to cross-shuffle
            int* indexes = stackalloc int[BatchesCount];
            for (int i = 0; i < BatchesCount; i++) indexes[i] = i;
            int n = BatchesCount;
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
            Parallel.For(0, BatchesCount / 2, Kernel).AssertCompleted();

            // Shuffle the main list
            n = Batches[BatchesCount - 1].X.GetLength(0) < BatchSize ? BatchesCount - 1 : BatchesCount; // Leave the odd batch in last position, if present
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
