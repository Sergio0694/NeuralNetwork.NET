using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Parameters
{
    public abstract class DatasetParametersBase
    {
        // TODO: add docs
        public (double[,] X, double[,] Y) Dataset { get; }

        protected DatasetParametersBase((double[,] X, double[,] Y) dataset)
        {
            if (dataset.X.GetLength(0) != dataset.Y.GetLength(0)) throw new ArgumentException(nameof(dataset), "The size of the input matrices isn't valid");
            Dataset = dataset;
        }
    }
}
