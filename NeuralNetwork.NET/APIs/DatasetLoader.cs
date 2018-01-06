using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using NeuralNetworkNET.SupervisedLearning.Optimization.Progress;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class with helper methods to easily create datasets to use to train and test a network
    /// </summary>
    public static class DatasetLoader
    {
        #region Training

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input collection, with the specified batch size
        /// </summary>
        /// <param name="data">The source collection to use to build the training dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training([NotNull] IEnumerable<(float[] X, float[] Y)> data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input collection, with the specified batch size
        /// </summary>
        /// <param name="data">The source collection to use to build the training dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training([NotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input matrices, with the specified batch size
        /// </summary>
        /// <param name="data">The source matrices to use to build the training dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training((float[,] X, float[,] Y) data, int size) => BatchesCollection.From(data, size);

        #endregion

        #region Test

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test([NotNull] IEnumerable<(float[] X, float[] Y)> data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null)
        {
            (float[] X, float[] Y)[] set = data.ToArray();
            float[,] 
                x = new float[set.Length, set[0].X.Length],
                y = new float[set.Length, set[0].Y.Length];
            Parallel.For(0, set.Length, i =>
            {
                set[i].X.AsSpan().CopyTo(x.Slice(i));
                set[i].Y.AsSpan().CopyTo(y.Slice(i));
            }).AssertCompleted();
            return new TestDataset((x, y), progress);
        }

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test([NotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null)
            => Test(data.AsParallel().Select(f => f()), progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test((float[,] X, float[,] Y) data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null) => new TestDataset(data, progress);

        #endregion
    }
}
