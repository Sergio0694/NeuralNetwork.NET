using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using NeuralNetworkNET.SupervisedLearning.Optimization.Progress;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A <see cref="delegate"/> that represents a factory that produces instances of a specific layer type, with user-defined parameters.
    /// This wrapper acts as an intemediary to streamline the user-side C# sintax when building up a new network structure, as all the input
    /// details for each layer will be automatically computed during the network setup.
    /// </summary>
    /// <param name="info">The tensor info for the inputs of the upcoming network layer</param>
    /// <remarks>It is also possible to invoke a <see cref="LayerFactory"/> instance just like any other <see cref="delegate"/> to immediately get an <see cref="INetworkLayer"/> value</remarks>
    [NotNull]
    public delegate INetworkLayer LayerFactory(TensorInfo info);

    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class NetworkManager
    {
        /// <summary>
        /// Creates a new network with a linear structure and the specified parameters
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> description</param>
        /// <param name="factories">A list of factories to create the different layers in the new network</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INeuralNetwork NewSequential(TensorInfo input, [NotNull, ItemNotNull] params LayerFactory[] factories)
        {
            IEnumerable<INetworkLayer> BuildLayers()
            {
                foreach (LayerFactory f in factories)
                {
                    INetworkLayer layer = f(input);
                    yield return layer;
                    input = layer.OutputInfo;
                }
            }
            return new NeuralNetwork(BuildLayers().ToArray());
        }

        #region Synchronous APIs

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A sequence of <see cref="ValueTuple{T1, T2}"/> tuples with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] INeuralNetwork network,
            IEnumerable<Func<(float[] X, float[] Y)>> trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            // Preliminary checks
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");

            // Start the training
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            return NetworkTrainer.TrainNetwork(network as NeuralNetwork, batches, epochs, dropout, algorithm, batchProgress, trainingProgress, validationParameters, testParameters, token);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A sequence of <see cref="ValueTuple{T1, T2}"/> tuples with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] INeuralNetwork network,
            IEnumerable<(float[] X, float[] Y)> trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            // Preliminary checks
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");

            // Start the training
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            return NetworkTrainer.TrainNetwork(network as NeuralNetwork, batches, epochs, dropout, algorithm, batchProgress, trainingProgress, validationParameters, testParameters, token);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A <see cref="ValueTuple{T1, T2}"/> tuple with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] INeuralNetwork network,
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            // Preliminary checks
            if (trainingSet.X.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (trainingSet.Y.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (trainingSet.X.GetLength(0) != trainingSet.Y.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (batchSize > trainingSet.X.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be less or equal than the number of training samples");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");

            // Start the training
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            return NetworkTrainer.TrainNetwork(network as NeuralNetwork, batches, epochs, dropout, algorithm, batchProgress, trainingProgress, validationParameters, testParameters, token);
        }

        #endregion

        #region Asynchronous APIs

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A sequence of <see cref="ValueTuple{T1, T2}"/> tuples with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            IEnumerable<Func<(float[] X, float[] Y)>> trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            return Task.Run(() => TrainNetwork(network, trainingSet, epochs, batchSize, algorithm, dropout, batchProgress, trainingProgress, validationParameters, testParameters, token), token);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A sequence of <see cref="ValueTuple{T1, T2}"/> tuples with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            IEnumerable<(float[] X, float[] Y)> trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            return Task.Run(() => TrainNetwork(network, trainingSet, epochs, batchSize, algorithm, dropout, batchProgress, trainingProgress, validationParameters, testParameters, token), token);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A <see cref="ValueTuple{T1, T2}"/> tuple with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            float dropout = 0,
            [CanBeNull] IProgress<BatchProgress> batchProgress = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress = null,
            [CanBeNull] ValidationParameters validationParameters = null,
            [CanBeNull] TestParameters testParameters = null,
            CancellationToken token = default)
        {
            return Task.Run(() => TrainNetwork(network, trainingSet, epochs, batchSize, algorithm, dropout, batchProgress, trainingProgress, validationParameters, testParameters, token), token);
        }

        #endregion

        #region Settings

        private static int _MaximumBatchSize = int.MaxValue;

        /// <summary>
        /// Gets or sets the maximum batch size (used to optimize the memory usage during validation/test processing)
        /// </summary>
        /// <remarks>Adjust this setting to the highest possible value according to the available RAM/VRAM and the size of the dataset. If the validation/test dataset has more
        /// samples than <see cref="MaximumBatchSize"/>, it will be automatically divided into batches so that it won't cause an <see cref="OutOfMemoryException"/> or other problems</remarks>
        public static int MaximumBatchSize
        {
            get => _MaximumBatchSize;
            set => _MaximumBatchSize = value >= 10 ? value : throw new ArgumentOutOfRangeException(nameof(MaximumBatchSize), "The maximum batch size must be at least equal to 10");
        }

        #endregion
    }
}
