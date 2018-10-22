using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

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
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training<TPixel>([NotNull] IEnumerable<(string X, float[] Y)> data, int size, ImageNormalizationMode normalization, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return BatchesCollection.From(modifiers.Length > 0 
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y))) 
                : data.Select<(string X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y)), size);
        }

        /// <summary>
        /// Creates a new <see cref="ITrainingDataset"/> instance to train a network from the input data, where each input sample is an image in a specified format
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="size">The desired dataset batch size</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITrainingDataset Training<TPixel>([NotNull] IEnumerable<(string X, Func<float[]> Y)> data, int size, ImageNormalizationMode normalization, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return BatchesCollection.From(modifiers.Length > 0 
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y()))) 
                : data.Select<(string X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y())), size);
        }

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
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation<TPixel>(
            [NotNull] IEnumerable<(string X, float[] Y)> data, float tolerance = 1e-2f, int epochs = 5, 
            ImageNormalizationMode normalization = ImageNormalizationMode.Sigmoid, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return Validation((modifiers.Length > 0
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y)))
                : data.Select<(string X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y))).AsParallel(), tolerance, epochs);
        }

        /// <summary>
        /// Creates a new <see cref="IValidationDataset"/> instance to validate a network accuracy from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="tolerance">The desired tolerance to test the network for convergence</param>
        /// <param name="epochs">The epochs interval to consider when testing the network for convergence</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static IValidationDataset Validation<TPixel>(
            [NotNull] IEnumerable<(string X, Func<float[]> Y)> data, float tolerance = 1e-2f, int epochs = 5, 
            ImageNormalizationMode normalization = ImageNormalizationMode.Sigmoid, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return Validation((modifiers.Length > 0
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y())))
                : data.Select<(string X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y()))).AsParallel(), tolerance, epochs);
        }

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
        public static ITestDataset Test([NotNull] IEnumerable<(float[] X, float[] Y)> data, [CanBeNull] Action<TrainingProgressEventArgs> progress = null)
            => new TestDataset(data.MergeLines(), progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset, where the samples will be extracted from the input <see cref="Func{TResult}"/> instances in parallel</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test([NotNull, ItemNotNull] IEnumerable<Func<(float[] X, float[] Y)>> data, [CanBeNull] Action<TrainingProgressEventArgs> progress = null)
            => Test(data.AsParallel().Select(f => f()), progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <param name="data">The source collection to use to build the test dataset</param>
        /// <param name="progress">The optional progress callback to use</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test((float[,] X, float[,] Y) data, [CanBeNull] Action<TrainingProgressEventArgs> progress = null) => new TestDataset(data, progress);

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a vector with the expected outputs</param>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test<TPixel>(
            [NotNull] IEnumerable<(string X, float[] Y)> data, [CanBeNull] Action<TrainingProgressEventArgs> progress = null
            , ImageNormalizationMode normalization = ImageNormalizationMode.Sigmoid, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return Test((modifiers.Length > 0
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y)))
                : data.Select<(string X, float[] Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y))).AsParallel(), progress);
        }

        /// <summary>
        /// Creates a new <see cref="ITestDataset"/> instance to test a network from the input collection
        /// </summary>
        /// <typeparam name="TPixel">The type of image pixels. It must be either <see cref="Alpha8"/>, <see cref="Rgb24"/>, <see cref="Argb32"/> or <see cref="Rgba32"/></typeparam>
        /// <param name="data">A list of <see cref="ValueTuple{T1, T2}"/> items, where the first element is the image path and the second is a <see cref="Func{TResult}"/> returning a vector with the expected outputs</param>
        /// <param name="progress">The optional progress callback to use</param>
        /// <param name="normalization">The desired image normalization mode to use when loading the images</param>
        /// <param name="modifiers">The optional <see cref="Action{T}"/> instances to use to modify the loaded image. If no modifiers are provided, each loaded image will not me tweaked. If one or more
        /// modifiers are passed to the method, a different image will be added to the dataset for each given modifier. This can be used to easily expand an image dataset.</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static ITestDataset Test<TPixel>(
            [NotNull] IEnumerable<(string X, Func<float[]> Y)> data, [CanBeNull] Action<TrainingProgressEventArgs> progress = null, 
            ImageNormalizationMode normalization = ImageNormalizationMode.Sigmoid, [NotNull, ItemNotNull] params Action<IImageProcessingContext<TPixel>>[] modifiers)
            where TPixel : struct, IPixel<TPixel>
        {
            return Test((modifiers.Length > 0
                ? data.SelectMany(xy => modifiers.Select<Action<IImageProcessingContext<TPixel>>, Func<(float[], float[])>>(f => () => (ImageLoader.Load(xy.X, normalization, f), xy.Y())))
                : data.Select<(string X, Func<float[]> Y), Func<(float[], float[])>>(xy => () => (ImageLoader.Load<TPixel>(xy.X, normalization, null), xy.Y()))).AsParallel(), progress);
        }

        #endregion
    }
}
