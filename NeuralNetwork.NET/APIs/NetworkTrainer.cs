using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace NeuralNetworkNET.APIs
{
    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class NetworkTrainer
    {
        /// <summary>
        /// Creates a new network with the specified parameters
        /// </summary>
        /// <param name="layers">The network layers to use</param>
        [PublicAPI]
        [Pure, NotNull]
        public static INeuralNetwork NewNetwork([NotNull, ItemNotNull] params INetworkLayer[] layers) => new NeuralNetwork(layers);

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A <see cref="ValueTuple{T1, T2}"/> tuple with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>
        /// <param name="eta">The desired learning rate for the stochastic gradient descent training</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="lambda">The optional L2 regularization value to scale down the weights during the gradient descent and avoid overfitting</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        /// <remarks>
        /// <para>The <paramref name="eta"/> value is divided by the <paramref name="batchSize"/> and indicates the rate at which the cost function is minimized</para>
        /// <para>The <paramref name="lambda"/> parameter (optional) depends on both <paramref name="eta"/> and the number of training samples and should be scaled accordingly</para>
        /// </remarks>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            ValidationParameters validationParameters = null,
            TestParameters testParameters = null,
            float eta = 0.1f, float dropout = 0, float lambda = 0, CancellationToken token = default)
        {
            // Preliminary checks
            if (!(network is NeuralNetwork localNet)) throw new ArgumentException(nameof(network), "Invalid network instance");
            if (trainingSet.X.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (trainingSet.Y.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (trainingSet.X.GetLength(0) != trainingSet.Y.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (batchSize > trainingSet.X.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be less or equal than the number of training samples");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");

            // Start the training
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            return Task.Run(() => localNet.StochasticGradientDescent(batches, epochs, validationParameters, testParameters, eta, dropout, lambda, token), token);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A list of <see cref="ValueTuple{T1, T2}"/> tuples with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>
        /// <param name="eta">The desired learning rate for the stochastic gradient descent training</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="lambda">The optional L2 regularization value to scale down the weights during the gradient descent and avoid overfitting</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        /// <remarks>
        /// <para>The <paramref name="eta"/> value is divided by the <paramref name="batchSize"/> and indicates the rate at which the cost function is minimized</para>
        /// <para>The <paramref name="lambda"/> parameter (optional) depends on both <paramref name="eta"/> and the number of training samples and should be scaled accordingly</para>
        /// </remarks>
        [PublicAPI]
        [NotNull, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingSessionResult> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            [NotNull] IReadOnlyList<(float[] X, float[] Y)> trainingSet,
            int epochs, int batchSize,
            ValidationParameters validationParameters = null,
            TestParameters testParameters = null,
            float eta = 0.1f, float dropout = 0, float lambda = 0, CancellationToken token = default)
        {
            // Preliminary checks
            if (!(network is NeuralNetwork localNet)) throw new ArgumentException(nameof(network), "Invalid network instance");
            if (trainingSet.Count == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (batchSize > trainingSet.Count) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be less or equal than the number of training samples");
            if (dropout < 0 || dropout >= 1) throw new ArgumentOutOfRangeException(nameof(dropout), "The dropout probability is invalid");

            // Start the training
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            return Task.Run(() => localNet.StochasticGradientDescent(batches, epochs, validationParameters, testParameters, eta, dropout, lambda, token), token);
        }
    }
}
