using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using NeuralNetworkNET.SupervisedLearning.Optimization.Abstract;

namespace NeuralNetworkNET.SupervisedLearning
{
    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class BackpropagationNetworkTrainer
    {
        #region Public APIs

        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="batchSize"></param>
        /// <param name="type">The type of learning algorithm to use to train the network</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <param name="neurons">The number of neurons in each network layer</param>
        [PublicAPI]
        [Pure, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            int? batchSize,
            LearningAlgorithmType type,
            CancellationToken token,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress,
            [NotNull] params int[] neurons)
        {
            return ComputeTrainedNetworkAsync(x, ys, batchSize ?? x.GetLength(0), type, token, null, progress, neurons);
        }

        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="batchSize"></param>
        /// <param name="network">The previous network to use as a starting point, to continue a training session</param>
        /// <param name="type">The type of learning algorithm to use to train the network</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <exception cref="ArgumentException">The input <see cref="INeuralNetwork"/> instance isn't valid</exception>
        [PublicAPI]
        [Pure, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            int? batchSize,
            [NotNull] INeuralNetwork network,
            LearningAlgorithmType type,
            CancellationToken token,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress)
        {
            double[] solution = (network as NeuralNetwork)?.Serialize() ?? throw new ArgumentException(nameof(network), "Invalid network instance");
            int[] neurons = new[] { network.InputLayerSize }.Concat(network.HiddenLayers).Concat(new[] { network.OutputLayerSize }).ToArray();
            return ComputeTrainedNetworkAsync(x, ys, batchSize ?? x.GetLength(0), type, token, solution, progress, neurons);
        }

        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="batchSize"></param>
        /// <param name="json">A JSON <see cref="String"/> representing a network to use to start the training</param>
        /// <param name="type">The type of learning algorithm to use to train the network</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <exception cref="ArgumentException">The input JSON <see cref="String"/> is invalid</exception>
        [PublicAPI]
        [Pure, ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            int? batchSize,
            [NotNull] String json,
            LearningAlgorithmType type,
            CancellationToken token,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress)
        {
            INeuralNetwork network = NeuralNetworkDeserializer.TryDeserialize(json);
            if (network == null) throw new ArgumentException("The input JSON file isn't valid");
            return ComputeTrainedNetworkAsync(x, ys, batchSize, network, type, token, progress);
        }

        #endregion

        // Private implementation
        private static async Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            int batchSize,
            LearningAlgorithmType type,
            CancellationToken token,
            [CanBeNull] double[] solution,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress,
            [NotNull] params int[] neurons)
        {
            // Preliminary checks
            if (x.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (x.GetLength(0) != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (neurons.Length < 2) throw new ArgumentOutOfRangeException("The network must have at least two layers");
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be a positive number");
            if (batchSize > x.GetLength(0)) throw new ArgumentOutOfRangeException(nameof(batchSize), "The batch size must be less or equal than the number of training samples");

            // Calculate the target network size
            double[] start = solution ?? NeuralNetwork.NewRandom(neurons).Serialize();

            // Prepare the batches
            int iteration = 1;
            TrainingBatch.BatchesCollection batches = TrainingBatch.BatchesCollection.FromDataset(x, ys, batchSize);

            // Get the optimization algorithm instance
            GradientOptimizationMethodBase optimizer;
            switch (type)
            {
                case LearningAlgorithmType.BoundedBFGS:
                case LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence:
                    optimizer = new BoundedBroydenFletcherGoldfarbShanno(start.Length);
                    break;
                case LearningAlgorithmType.GradientDescent:
                    optimizer = new GradientDescent { NumberOfVariables = start.Length };
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), "Unsupported optimization method");
            }
            optimizer.Solution = start;
            optimizer.Token = token;
            optimizer.Function = CostFunction;
            optimizer.Gradient = GradientFunction;

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                double cost = network.CalculateCost(x, ys);
                if (!double.IsNaN(cost))
                {
                    progress?.Report(new BackpropagationProgressEventArgs(
                        () => NeuralNetwork.Deserialize(optimizer.Solution, neurons), iteration++, cost));
                }
                return cost;
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                TrainingBatch pick = batches.Next();
                return network.ComputeGradient(pick.X, pick.Y);
            }

            // Minimize the cost function
            await Task.Run(() => optimizer.Minimize(), token);

            // Check if second optimization is pending
            if (type == LearningAlgorithmType.BoundedBFGSWithGradientDescentOnFirstConvergence && !token.IsCancellationRequested)
            {
                // Reinitialize the optimizer
                double[] partial = optimizer.Solution;
                optimizer = new GradientDescent
                {
                    NumberOfVariables = start.Length,
                    Solution = partial,
                    Token = token,
                    Function = CostFunction,
                    Gradient = GradientFunction
                };

                // Optimize again
                await Task.Run(() => optimizer.Minimize(), token);
            }

            // Return the result network
            return NeuralNetwork.Deserialize(optimizer.Solution, neurons);
        }
    }
}
