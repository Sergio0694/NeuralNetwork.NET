using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.SupervisedLearning.Parameters
{
    /// <summary>
    /// A base class for an optional dataset to use in a training session
    /// </summary>
    internal abstract class DatasetBase : IDataset
    {
        /// <summary>
        /// Gets the raw content of the current dataset
        /// </summary>
        public (float[,] X, float[,] Y) Dataset { get; }

        #region Interface

        /// <inheritdoc/>
        public int Count => Dataset.X.GetLength(0);

        /// <inheritdoc/>
        public int InputFeatures => Dataset.X.GetLength(1);

        /// <inheritdoc/>
        public int OutputFeatures => Dataset.Y.GetLength(1);

        /// <inheritdoc/>
        public long ByteSize => sizeof(float) * Count * (InputFeatures + OutputFeatures);

        /// <inheritdoc/>
        public DatasetSample this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (i < 0 || i > Count - 1) throw new ArgumentOutOfRangeException(nameof(i), "The target index is not valid");
                return new DatasetSample(Dataset.X.Slice(i), Dataset.Y.Slice(i));
            }
        }

        /// <inheritdoc/>
        public unsafe int Id
        {
            get
            {
                int[] temp = new int[Count];
                int
                    wx = Dataset.X.GetLength(1),
                    wy = Dataset.Y.GetLength(1);
                fixed (float* px0 = Dataset.X, py0 = Dataset.Y)
                {
                    float* px = px0, py = py0;
                    Parallel.For(0, Count, i =>
                    {
                        temp[i] = new Span<float>(px + i * wx, wx).GetContentHashCode() ^ new Span<float>(py + i * wy, wy).GetContentHashCode();
                    }).AssertCompleted();
                }
                Array.Sort(temp);
                return temp.AsSpan().GetContentHashCode();
            }
        }

        #endregion

        protected internal DatasetBase((float[,] X, float[,] Y) dataset)
        {
            if (dataset.X.GetLength(0) != dataset.Y.GetLength(0)) throw new ArgumentException("The size of the input matrices isn't valid", nameof(dataset));
            Dataset = dataset;
        }
    }
}
