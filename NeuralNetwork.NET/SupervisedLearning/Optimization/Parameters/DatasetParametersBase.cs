using System;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    /// <summary>
    /// A base class for an optional dataset to use in a training session
    /// </summary>
    public abstract class DatasetParametersBase
    {
        /// <summary>
        /// Gets the current dataset
        /// </summary>
        public (float[,] X, float[,] Y) Dataset { get; }

        protected DatasetParametersBase((float[,] X, float[,] Y) dataset)
        {
            if (dataset.X.GetLength(0) != dataset.Y.GetLength(0)) throw new ArgumentException(nameof(dataset), "The size of the input matrices isn't valid");
            Dataset = dataset;
        }
    }
}
