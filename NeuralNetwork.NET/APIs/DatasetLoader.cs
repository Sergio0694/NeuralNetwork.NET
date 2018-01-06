using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.SupervisedLearning.Data;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class with helper methods to easily create datasets to use to train and test a network
    /// </summary>
    public static class DatasetLoader
    {
        /// <summary>
        /// Creates a new <see cref="IDataset"/> instance to train a network from the input collection, with the specified batch size
        /// </summary>
        /// <param name="data">The source collection to use to build the dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IDataset Training([NotNull] IEnumerable<(float[] X, float[] Y)> data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="IDataset"/> instance to train a network from the input collection, with the specified batch size
        /// </summary>
        /// <param name="data">The source collection to use to build the dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IDataset Training([NotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="IDataset"/> instance to train a network from the input matrices, with the specified batch size
        /// </summary>
        /// <param name="data">The source matrices to use to build the dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IDataset Training((float[,] X, float[,] Y) data, int size) => BatchesCollection.From(data, size);
    }
}
