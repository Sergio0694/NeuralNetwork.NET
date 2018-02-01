using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Delegates;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Graph;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using NeuralNetworkNET.SupervisedLearning.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.APIs
{
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
            return new SequentialNetwork(factories.Aggregate(new List<INetworkLayer>(), (l, f) =>
            {
                INetworkLayer layer = f(input);
                input = layer.OutputInfo;
                l.Add(layer);
                return l;
            }).ToArray());
        }

        /// <summary>
        /// Creates a computational graph network with a custom structure
        /// </summary>
        /// <param name="input">The input <see cref="TensorInfo"/> description</param>
        /// <param name="builder">An <see cref="Action{T}"/> used to build the graph from the input <see cref="NodeBuilder"/> node</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INeuralNetwork NewGraph(TensorInfo input, [NotNull] Action<NodeBuilder> builder)
        {
            NodeBuilder root = NodeBuilder.Input();
            builder(root);
            ComputationGraph graph = ComputationGraph.New(input, root);
            return new ComputationGraphNetwork(graph);
        }

        #region Training APIs

        /// <summary>
        /// Gets whether or not a neural network is currently being trained
        /// </summary>
        public static bool TrainingInProgress { get; private set; }

        /// <summary>
        /// Trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="dataset">The <see cref="ITrainingDataset"/> instance to use to train the network</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="Enums.LayerType.FullyConnected"/> layer</param>
        /// <param name="batchCallback">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingCallback">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationDataset">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testDataset">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] INeuralNetwork network,
            [NotNull] ITrainingDataset dataset,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            int epochs, float dropout = 0,
            [CanBeNull] Action<BatchProgress> batchCallback = null,
            [CanBeNull] Action<TrainingProgressEventArgs> trainingCallback = null,
            [CanBeNull] IValidationDataset validationDataset = null,
            [CanBeNull] ITestDataset testDataset = null,
            CancellationToken token = default)
        {
            return TrainNetworkCore(network, dataset, algorithm, epochs, dropout, batchCallback.AsIProgress(), trainingCallback.AsIProgress(), validationDataset, testDataset, token);
        }

        /// <summary>
        /// Trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="dataset">The <see cref="ITrainingDataset"/> instance to use to train the network</param>
        /// <param name="algorithm">The desired training algorithm to use</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="Enums.LayerType.FullyConnected"/> layer</param>
        /// <param name="batchCallback">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingCallback">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationDataset">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testDataset">The optional test dataset to use to monitor the current generalized training progress</param>       
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            [NotNull] ITrainingDataset dataset,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            int epochs, float dropout = 0,
            [CanBeNull] Action<BatchProgress> batchCallback = null,
            [CanBeNull] Action<TrainingProgressEventArgs> trainingCallback = null,
            [CanBeNull] IValidationDataset validationDataset = null,
            [CanBeNull] ITestDataset testDataset = null,
            CancellationToken token = default)
        {
            IProgress<BatchProgress> batchProgress = batchCallback.AsIProgress();
            IProgress<TrainingProgressEventArgs> trainingProgress = trainingCallback.AsIProgress(); // Capture the synchronization contexts
            return Task.Run(() => TrainNetworkCore(network, dataset, algorithm, epochs, dropout, batchProgress, trainingProgress, validationDataset, testDataset, token), token);
        }

        #endregion

        // Core trainer method with additional checks
        [NotNull]
        private static TrainingSessionResult TrainNetworkCore(
            [NotNull] INeuralNetwork network,
            [NotNull] ITrainingDataset dataset,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            int epochs, float dropout,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] IValidationDataset validationDataset,
            [CanBeNull] ITestDataset testDataset,
            CancellationToken token)
        {
            // Preliminary checks
            if (epochs < 1) throw new ArgumentOutOfRangeException(nameof(epochs), "The number of epochs must at be at least equal to 1");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");
            if (validationDataset != null && (validationDataset.InputFeatures != dataset.InputFeatures || validationDataset.OutputFeatures != dataset.OutputFeatures))
                throw new ArgumentException("The validation dataset doesn't match the training dataset", nameof(validationDataset));
            if (testDataset != null && (testDataset.InputFeatures != dataset.InputFeatures || testDataset.OutputFeatures != dataset.OutputFeatures))
                throw new ArgumentException("The test dataset doesn't match the training dataset", nameof(testDataset));
            if (dataset.InputFeatures != network.InputInfo.Size || dataset.OutputFeatures != network.OutputInfo.Size)
                throw new ArgumentException("The input dataset doesn't match the number of input and output features for the current network", nameof(dataset));

            // Start the training
            TrainingInProgress = TrainingInProgress
                ? throw new InvalidOperationException("Can't train two networks at the same time") // This would cause problems with cuDNN
                : true;
            TrainingSessionResult result = NetworkTrainer.TrainNetwork(
                network as NeuralNetworkBase ?? throw new ArgumentException("The input network instance isn't valid", nameof(network)), 
                dataset as BatchesCollection ?? throw new ArgumentException("The input dataset instance isn't valid", nameof(dataset)),
                epochs, dropout, algorithm, batchProgress, trainingProgress, 
                validationDataset as ValidationDataset,
                testDataset as TestDataset,
                token);
            TrainingInProgress = false;
            return result;
        }
    }
}
