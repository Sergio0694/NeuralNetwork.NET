using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
        /// <param name="data">The source collection to use to build the training dataset, where the samples will be extracted from the input <see cref="Func{TResult}"/> instances in parallel</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training([NotNull, ItemNotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input matrices, with the specified batch size
        /// </summary>
        /// <param name="data">The source matrices to use to build the training dataset</param>
        /// <param name="size">The desired dataset batch size</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training((float[,] X, float[,] Y) data, int size) => BatchesCollection.From(data, size);

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input data, where each input sample is an image in a specified format
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training<TPixel>([NotNull] IEnumerable<(String X, float[] Y)> data, int size, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => BatchesCollection.From(data.Select<(String X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y)), size);

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input data, where each input sample is an image in a specified format
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training<TPixel>([NotNull] IEnumerable<(String X, Func<float[]> Y)> data, int size, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => BatchesCollection.From(data.Select<(String X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y())), size);

        #endregion

        #region Validation

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the validation dataset</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation([NotNull] IEnumerable<(float[] X, float[] Y)> data, float tolerance = 1e-2f, int epochs = 5)
            => new ValidationDataset(data.MergeLines(), tolerance, epochs);

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the validation dataset, where the samples will be extracted from the input <see cref="Func{TResult}"/> instances in parallel</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation([NotNull, ItemNotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, float tolerance = 1e-2f, int epochs = 5)
            => Validation(data.AsParallel().Select(f => f()), tolerance, epochs);

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the validation dataset</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation((float[,] X, float[,] Y) data, float tolerance = 1e-2f, int epochs = 5) => new ValidationDataset(data, tolerance, epochs);

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation<TPixel>([NotNull] IEnumerable<(String X, float[] Y)> data, float tolerance = 1e-2f, int epochs = 5, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => Validation(data.Select<(String X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y)).AsParallel(), tolerance, epochs);

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation<TPixel>([NotNull] IEnumerable<(String X, Func<float[]> Y)> data, float tolerance = 1e-2f, int epochs = 5, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => Validation(data.Select<(String X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y())).AsParallel(), tolerance, epochs);

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
            => new TestDataset(data.MergeLines(), progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset, where the samples will be extracted from the input <see cref="Func{TResult}"/> instances in parallel</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test([NotNull, ItemNotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null)
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

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test<TPixel>([NotNull] IEnumerable<(String X, float[] Y)> data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => Test(data.Select<(String X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y)).AsParallel(), progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/> or <see cref="Argb32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="modify">An optional <see cref="Action{T}"/> to modify each sample image when loading the dataset</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test<TPixel>([NotNull] IEnumerable<(String X, Func<float[]> Y)> data, [CanBeNull] IProgress<TrainingProgressEventArgs> progress = null, [CanBeNull] Action<IImageProcessingContext<TPixel>> modify = null)
            where TPixel : struct, IPixel<TPixel>
            => Test(data.Select<(String X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load(xy.X, modify), xy.Y())).AsParallel(), progress);

        #endregion
    }
}
