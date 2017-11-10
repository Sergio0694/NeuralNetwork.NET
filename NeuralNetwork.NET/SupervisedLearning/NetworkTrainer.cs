using System;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
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
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="batchSize"></param>
        /// <param name="learningType">The type of learning algorithm to use to train the network</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <param name="layers">The network layers to create</param>
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
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="batchSize"></param>
        /// <param name="network">The previous network to use as a starting point, to continue a training session</param>
        /// <param name="learningType">The type of learning algorithm to use to train the network</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <exception cref="ArgumentException">The input <see cref="INeuralNetwork"/> instance isn't valid</exception>
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
