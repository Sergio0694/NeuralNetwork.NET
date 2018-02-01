using System;
using System.Collections.Generic;
using System.Threading;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Services;
using NeuralNetworkNET.SupervisedLearning.Algorithms.Info;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;
using NeuralNetworkNET.SupervisedLearning.Trackers;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    /// <summary>
    /// A static class that contains various network optimization algorithms
    /// </summary>
    internal static class NetworkTrainer
    {
        /// <summary>
        /// Trains the target <see cref="NeuralNetworkBase"/> instance with the given parameters and data
        /// </summary>
        /// <param name="network">The target <see cref="NeuralNetworkBase"/> to train</param>
        /// <param name="batches">The training baatches for the current session</param>
        /// <param name="epochs">The desired number of training epochs to run</param>
        /// <param name="dropout">Indicates the dropout probability for neurons in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="algorithm">The info on the training algorithm to use</param>
        /// <param name="batchProgress">An optional callback to monitor the training progress (in terms of dataset completion)</param>
        /// <param name="trainingProgress">An optional progress callback to monitor progress on the training dataset (in terms of classification performance)</param>
        /// <param name="validationDataset">The optional <see cref="ValidationDataset"/> instance (used for early-stopping)</param>
        /// <param name="testDataset">The optional <see cref="TestDataset"/> instance (used to monitor the training progress)</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
        [NotNull]
        public static TrainingSessionResult TrainNetwork(
            [NotNull] NeuralNetworkBase network, [NotNull] BatchesCollection batches,
            int epochs, float dropout,
            [NotNull] ITrainingAlgorithmInfo algorithm,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
            CancellationToken token)
        {
            SharedEventsService.TrainingStarting.Raise();
            WeightsUpdater optimizer;
            switch (algorithm)
            {
                /* =================
                 * Optimization
                 * =================
                 * The right optimizer is selected here, and the capatured closure for each of them also contains local temporary data, if needed.
                 * In this case, the temporary data is managed, so that it will automatically be disposed by the GC and there won't be the need to use
                 * another callback when the training stops to handle the cleanup of unmanaged resources. */
                case MomentumInfo momentum:
                    optimizer = WeightsUpdaters.Momentum(momentum, network);
                    break;
                case StochasticGradientDescentInfo sgd:
                    optimizer = WeightsUpdaters.StochasticGradientDescent(sgd);
                    break;
                case AdaGradInfo adagrad:
                    optimizer = WeightsUpdaters.AdaGrad(adagrad, network);
                    break;
                case AdaDeltaInfo adadelta:
                    optimizer = WeightsUpdaters.AdaDelta(adadelta, network);
                    break;
                case AdamInfo adam:
                    optimizer = WeightsUpdaters.Adam(adam, network);
                    break;
                case AdaMaxInfo adamax:
                    optimizer = WeightsUpdaters.AdaMax(adamax, network);
                    break;
                case RMSPropInfo rms:
                    optimizer = WeightsUpdaters.RMSProp(rms, network);
                    break;
                default:
                    throw new ArgumentException("The input training algorithm type is not supported");
            }
            return Optimize(network, batches, epochs, dropout, optimizer, batchProgress, trainingProgress, validationDataset, testDataset, token);
        }

        /// <summary>
        /// Gets whether or not a neural network is currently processing the training samples through backpropagation (as opposed to evaluating them)
        /// </summary>
        public static bool BackpropagationInProgress { get; private set; }

        /// <summary>
        /// Trains the target <see cref="SequentialNetwork"/> using the input algorithm
        /// </summary>
        [NotNull]
        private static TrainingSessionResult Optimize(
            NeuralNetworkBase network,
            BatchesCollection miniBatches,
            int epochs, float dropout,
            [NotNull] WeightsUpdater updater,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
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
            RelativeConvergence convergence = validationDataset == null
                ? null
                : new RelativeConvergence(validationDataset.Tolerance, validationDataset.EpochsInterval);

            // Optional batch monitor
            BatchProgressMonitor batchMonitor = batchProgress == null ? null : new BatchProgressMonitor(miniBatches.Count, batchProgress);

            // Create the training batches
            for (int i = 0; i < epochs; i++)
            {
                // Shuffle the training set
                miniBatches.CrossShuffle();

                // Gradient descent over the current batches
                BackpropagationInProgress = true;
                for (int j = 0; j < miniBatches.BatchesCount; j++)
                {
                    if (token.IsCancellationRequested)
                    {
                        BackpropagationInProgress = false;
                        return PrepareResult(TrainingStopReason.TrainingCanceled, i);
                    }
                    network.Backpropagate(miniBatches.Batches[j], dropout, updater);
                    batchMonitor?.NotifyCompletedBatch(miniBatches.Batches[j].X.GetLength(0));
                }
                BackpropagationInProgress = false;
                batchMonitor?.Reset();
                if (network.IsInNumericOverflow) return PrepareResult(TrainingStopReason.NumericOverflow, i);

                // Check the training progress
                if (trainingProgress != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(miniBatches);
                    trainingProgress.Report(new TrainingProgressEventArgs(i + 1, cost, accuracy));
                }

                // Check the validation dataset
                if (convergence != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(validationDataset.Dataset);
                    validationReports.Add(new DatasetEvaluationResult(cost, accuracy));
                    convergence.Value = accuracy;
                    if (convergence.HasConverged) return PrepareResult(TrainingStopReason.EarlyStopping, i);
                }

                // Report progress if necessary
                if (testDataset != null)
                {
                    (float cost, _, float accuracy) = network.Evaluate(testDataset.Dataset);
                    testReports.Add(new DatasetEvaluationResult(cost, accuracy));
                    testDataset.ThreadSafeProgressCallback?.Report(new TrainingProgressEventArgs(i + 1, cost, accuracy));
                }
            }
            return PrepareResult(TrainingStopReason.EpochsCompleted, epochs);
        }
    }
}
