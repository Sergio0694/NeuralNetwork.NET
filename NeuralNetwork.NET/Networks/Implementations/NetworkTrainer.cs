using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.DependencyInjections;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Algorithms.Info;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using NeuralNetworkNET.SupervisedLearning.Optimization.Progress;
using NeuralNetworkNET.SupervisedLearning.Progress;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A static class that contains various network optimization algorithms
    /// </summary>
    internal static class NetworkTrainer
    {
        /// <summary>
        /// Trains the target <see cref="NeuralNetwork"/> with the given parameters and data
        /// </summary>
        /// <param name="network">The target <see cref="NeuralNetwork"/> to train</param>
        /// <param name="batches">The training baatches for the current session</param>
        /// <param name="epochs">The desired number of training epochs to run</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="algorithm">The info on the training algorithm to use</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationParameters">The optional <see cref="ValidationParameters"/> instance (used for early-stopping)</param>
        /// <param name="testParameters">The optional <see cref="TestParameters"/> instance (used to monitor the training progress)</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [NotNull]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] NeuralNetwork network, [NotNull] BatchesCollection batches,
            int epochs, float dropout,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationParameters validationParameters,
            [CanBeNull] TestParameters testParameters,
            CancellationToken token)
        {
            LibraryRuntimeHelper.SynchronizeContext?.Invoke();
            switch (algorithm)
            {
                case StochasticGradientDescentInfo sgd:
                    return StochasticGradientDescent(network, batches, epochs, dropout, sgd.Eta, sgd.Lambda, batchProgress, trainingProgress, validationParameters, testParameters, token);
                case AdadeltaInfo adadelta:
                    return Adadelta(network, batches, epochs, dropout, adadelta.Rho, adadelta.Epsilon, adadelta.L2, batchProgress, trainingProgress, validationParameters, testParameters, token);
                default:
                    throw new ArgumentException("The input training algorithm type is not supported");
            }
        }

        #region Optimization algorithms

        /// <summary>
        /// Trains the target <see cref="NeuralNetwork"/> using the gradient descent algorithm
        /// </summary>
        [NotNull]
        private static TrainingSessionResult StochasticGradientDescent(
            NeuralNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float eta, float lambda,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationParameters validationParameters,
            [CanBeNull] TestParameters testParameters,
            CancellationToken token)
        {
            // Plain SGD weights update
            unsafe void Minimize(int i, in Tensor dJdw, in Tensor dJdb, int samples, WeightedLayerBase layer)
            {
                // Tweak the weights
                float
                    alpha = eta / samples,
                    l2Factor = eta * lambda / samples;
                fixed (float* pw = layer.Weights)
                {
                    float* pdj = dJdw;
                    int w = layer.Weights.Length;
                    for (int x = 0; x < w; x++)
                        pw[x] -= l2Factor * pw[x] + alpha * pdj[x];
                }

                // Tweak the biases of the lth layer
                fixed (float* pb = layer.Biases)
                {
                    float* pdj = dJdb;
                    int w = layer.Biases.Length;
                    for (int x = 0; x < w; x++)
                        pb[x] -= alpha * pdj[x];
                }
            }

            return Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationParameters, testParameters, token);
        }

        /// <summary>
        /// Trains the target <see cref="NeuralNetwork"/> using the Adadelta algorithm
        /// </summary>
        [NotNull]
        private static unsafe TrainingSessionResult Adadelta(
            NeuralNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float rho, float epsilon, float l2Factor,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationParameters validationParameters,
            [CanBeNull] TestParameters testParameters,
            CancellationToken token)
        {
            // Initialize Adadelta parameters
            Tensor*
                egSquaredW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                eDeltaxSquaredW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                egSquaredB = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                eDeltaxSquaredB = stackalloc Tensor[network.WeightedLayersIndexes.Length];
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                WeightedLayerBase layer = network._Layers[network.WeightedLayersIndexes[i]].To<NetworkLayerBase, WeightedLayerBase>();
                Tensor.NewZeroed(1, layer.Weights.Length, out egSquaredW[i]);
                Tensor.NewZeroed(1, layer.Weights.Length, out eDeltaxSquaredW[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out egSquaredB[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out eDeltaxSquaredB[i]);
            }

            // Adadelta update for weights and biases
            void Minimize(int i, in Tensor dJdw, in Tensor dJdb, int samples, WeightedLayerBase layer)
            {
                fixed (float* pw = layer.Weights)
                {
                    float*
                        pdj = dJdw,
                        egSqrt = egSquaredW[i],
                        eDSqrtx = eDeltaxSquaredW[i];
                    int w = layer.Weights.Length;
                    for (int x = 0; x < w; x++)
                    {
                        float gt = pdj[x];
                        egSqrt[x] = rho * egSqrt[x] + (1 - rho) * gt * gt;
                        float
                            rmsDx_1 = (float)Math.Sqrt(eDSqrtx[x] + epsilon),
                            rmsGt = (float)Math.Sqrt(egSqrt[x] + epsilon),
                            dx = -(rmsDx_1 / rmsGt) * gt;
                        eDSqrtx[x] = rho * eDSqrtx[x] + (1 - rho) * dx * dx;
                        pw[x] += dx - l2Factor * pw[x];
                    }
                }

                // Tweak the biases of the lth layer
                fixed (float* pb = layer.Biases)
                {
                    float*
                        pdj = dJdb,
                        egSqrt = egSquaredB[i],
                        eDSqrtb = eDeltaxSquaredB[i];
                    int w = layer.Biases.Length;
                    for (int b = 0; b < w; b++)
                    {
                        float gt = pdj[b];
                        egSqrt[b] = rho * egSqrt[b] + (1 - rho) * gt * gt;
                        float
                            rmsDx_1 = (float)Math.Sqrt(eDSqrtb[b] + epsilon),
                            rmsGt = (float)Math.Sqrt(egSqrt[b] + epsilon),
                            db = -(rmsDx_1 / rmsGt) * gt;
                        eDSqrtb[b] = rho * eDSqrtb[b] + (1 - rho) * db * db;
                        pb[b] += db - l2Factor * pb[b];
                    }
                }
            }

            TrainingSessionResult result = Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationParameters, testParameters, token);

            // Cleanup
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                WeightedLayerBase layer = network._Layers[network.WeightedLayersIndexes[i]].To<NetworkLayerBase, WeightedLayerBase>();
                egSquaredW[i].Free();
                eDeltaxSquaredW[i].Free();
                egSquaredB[i].Free();
                eDeltaxSquaredB[i].Free();
            }
            return result;
        }

        #endregion

        #region Core optimization

        /// <summary>
        /// Trains the target <see cref="NeuralNetwork"/> using the input algorithm
        /// </summary>
        [NotNull]
        private static TrainingSessionResult Optimize(
            NeuralNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout,
            [NotNull] WeightsUpdater updater,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationParameters validationParameters,
            [CanBeNull] TestParameters testParameters,
            CancellationToken token)
        {
            // Setup
            DateTime startTime = DateTime.Now;
            List<DatasetEvaluationResult>
                validationReports = new List<DatasetEvaluationResult>(),
                testReports = new List<DatasetEvaluationResult>();
            TrainingSessionResult PrepareResult(TrainingStopReason reason, int loops)
            {
                return new TrainingSessionResult(reason, loops, DateTime.Now.Subtract(startTime).RoundToSeconds(), validationReports, testReports);
            }

            // Convergence manager for the validation dataset
            RelativeConvergence convergence = validationParameters == null
                ? null
                : new RelativeConvergence(validationParameters.Tolerance, validationParameters.EpochsInterval);

            // Optional batch monitor
            BatchProgressMonitor batchMonitor = batchProgress == null ? null : new BatchProgressMonitor(miniBatches.Samples, batchProgress);

            // Create the training batches
            for (int i = 0; i < epochs; i++)
            {
                // Shuffle the training set
                miniBatches.CrossShuffle();

                // Gradient descent over the current batches
                for (int j = 0; j < miniBatches.Count; j++)
                {
                    if (token.IsCancellationRequested) return PrepareResult(TrainingStopReason.TrainingCanceled, i);
                    network.Backpropagate(miniBatches.Batches[j], dropout, updater);
                    batchMonitor?.NotifyCompletedBatch(miniBatches.Batches[j].X.GetLength(0));
                }
                batchMonitor?.Reset();

                // Check for overflows
                if (!Parallel.For(0, network._Layers.Length, (j, state) =>
                {
                    if (network._Layers[j] is WeightedLayerBase layer && !layer.ValidateWeights()) state.Break();
                }).IsCompleted) return PrepareResult(TrainingStopReason.NumericOverflow, i);

                // Check the training progress
                if (trainingProgress != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(miniBatches);
                    trainingProgress.Report(new BackpropagationProgressEventArgs(i + 1, cost, accuracy));
                }

                // Check the validation dataset
                if (convergence != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(validationParameters.Dataset);
                    validationReports.Add(new DatasetEvaluationResult(cost, accuracy));
                    convergence.Value = accuracy;
                    if (convergence.HasConverged) return PrepareResult(TrainingStopReason.EarlyStopping, i);
                }

                // Report progress if necessary
                if (testParameters != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(testParameters.Dataset);
                    testReports.Add(new DatasetEvaluationResult(cost, accuracy));
                    testParameters.ProgressCallback.Report(new BackpropagationProgressEventArgs(i + 1, cost, accuracy));
                }
            }
            return PrepareResult(TrainingStopReason.EpochsCompleted, epochs);
        }

        #endregion
    }
}
