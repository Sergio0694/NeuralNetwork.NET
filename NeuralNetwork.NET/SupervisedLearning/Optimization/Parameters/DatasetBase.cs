using System;
using System.Runtime.CompilerServices;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    /// <summary>
    /// A base class for an optional dataset to use in a training session
    /// </summary>
    public abstract class DatasetBase : IDataset
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

        #endregion

        protected internal DatasetBase((float[,] X, float[,] Y) dataset)
        {
            if (dataset.X.GetLength(0) != dataset.Y.GetLength(0)) throw new ArgumentException(nameof(dataset), "The size of the input matrices isn't valid");
            Dataset = dataset;
        }
    }
}
