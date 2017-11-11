using System;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;

namespace NeuralNetworkNET.SupervisedLearning
{
    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class NetworkTrainer
    {
        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="trainingSet">A <see cref="ValueTuple{T1, T2}"/> with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>
        /// <param name="eta">The desired learning rate for the stochastic gradient descent training</param>
        /// <param name="lambda">The optional L2 regularization value to scale down the weights during the gradient descent and avoid overfitting</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        /// <param name="layers">The desired layers of the network to create and train</param>
        /// <remarks>
        /// <para>The <paramref name="eta"/> value is divided by the <paramref name="batchSize"/> and indicates the rate at which the cost function is minimized</para>
        /// <para>The <paramref name="lambda"/> parameter (optional) depends on both <paramref name="eta"/> and the number of training samples and should be scaled accordingly</para>
        /// </remarks>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public static async Task<(INeuralNetwork Network, TrainingStopReason StopReason)> TrainNetworkAsync(
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            ValidationParameters validationParameters = null,
            TestParameters testParameters = null,
            float eta = 0.1f, float lambda = 0f, CancellationToken token = default,
            [NotNull, ItemNotNull] params NetworkLayer[] layers)
        {
            NeuralNetwork network = NeuralNetwork.NewRandom(layers);
            TrainingStopReason result = await TrainNetworkAsync(
                network, trainingSet, epochs, batchSize, validationParameters, testParameters, eta, lambda, token);
            return (network, result);
        }

        /// <summary>
        /// Generates and trains a neural network with the given parameters
        /// </summary>
        /// <param name="network">The existing <see cref="INeuralNetwork"/> to train with the given dataset(s)</param>
        /// <param name="trainingSet">A <see cref="ValueTuple{T1, T2}"/> with the training samples and expected results</param>
        /// <param name="epochs">The number of epochs to run with the training data</param>
        /// <param name="batchSize">The size of each training batch that the dataset will be divided into</param>
        /// <param name="validationParameters">An optional dataset used to check for convergence and avoid overfitting</param>
        /// <param name="testParameters">The optional test dataset to use to monitor the current generalized training progress</param>
        /// <param name="eta">The desired learning rate for the stochastic gradient descent training</param>
        /// <param name="lambda">The optional L2 regularization value to scale down the weights during the gradient descent and avoid overfitting</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        /// <remarks>
        /// <para>The <paramref name="eta"/> value is divided by the <paramref name="batchSize"/> and indicates the rate at which the cost function is minimized</para>
        /// <para>The <paramref name="lambda"/> parameter (optional) depends on both <paramref name="eta"/> and the number of training samples and should be scaled accordingly</para>
        /// </remarks>
        [PublicAPI]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<TrainingStopReason> TrainNetworkAsync(
            [NotNull] INeuralNetwork network,
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            ValidationParameters validationParameters = null,
            TestParameters testParameters = null,
            float eta = 0.1f, float lambda = 0f, CancellationToken token = default)
        {
            // Preliminary checks
            if (!(network is NeuralNetwork localNet)) throw new ArgumentException(nameof(network), "Invalid network instance");
            if (trainingSet.X.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (trainingSet.Y.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (trainingSet.X.GetLength(0) != trainingSet.Y.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (batchSize > trainingSet.X.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be less or equal than the number of training samples");

            // Start the training
            return Task.Run(() => localNet.StochasticGradientDescent(
                trainingSet, epochs, batchSize, validationParameters, testParameters, eta, lambda, token), token);
        }
    }
}
