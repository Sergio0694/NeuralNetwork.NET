using System;
using System.Threading;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Parameters;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    /// <inheritdoc cref="NetworkTrainer"/>
    internal static partial class NetworkTrainer
    {
        // Classic SGD algorithm
        [NotNull]
        private static TrainingSessionResult StochasticGradientDescent(
            SequentialNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float eta, float lambda,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
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

            return Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationDataset, testDataset, token);
        }

        // Adadelta method
        [NotNull]
        private static unsafe TrainingSessionResult Adadelta(
            SequentialNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float rho, float epsilon, float l2Factor,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
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

            TrainingSessionResult result = Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationDataset, testDataset, token);

            // Cleanup
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                egSquaredW[i].Free();
                eDeltaxSquaredW[i].Free();
                egSquaredB[i].Free();
                eDeltaxSquaredB[i].Free();
            }
            return result;
        }

        // Adam method
        [NotNull]
        private static unsafe TrainingSessionResult Adam(
            SequentialNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float eta, float beta1, float beta2, float epsilon,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
            CancellationToken token)
        {
            // Initialize Adadelta parameters
            Tensor*
                mW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                vW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                mB = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                vB = stackalloc Tensor[network.WeightedLayersIndexes.Length];
            Tensor.New(1, network.WeightedLayersIndexes.Length, out Tensor beta1t);
            Tensor.New(1, network.WeightedLayersIndexes.Length, out Tensor beta2t);
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                WeightedLayerBase layer = network._Layers[network.WeightedLayersIndexes[i]].To<NetworkLayerBase, WeightedLayerBase>();
                Tensor.NewZeroed(1, layer.Weights.Length, out mW[i]);
                Tensor.NewZeroed(1, layer.Weights.Length, out vW[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out mB[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out vB[i]);
                beta1t[i] = beta1;
                beta2t[i] = beta2;
            }

            // Adadelta update for weights and biases
            void Minimize(int i, in Tensor dJdw, in Tensor dJdb, int samples, WeightedLayerBase layer)
            {
                // Alpha at timestep t
                float alphat = eta * (float)Math.Sqrt(1 - beta2t[i]) / (1 - beta1t[i]);
                beta1t[i] *= beta1;
                beta2t[i] *= beta2;

                // Weights
                fixed (float* pw = layer.Weights)
                {
                    float*
                        pdJ = dJdw,
                        pm = mW[i],
                        pv = vW[i];
                    int w = layer.Weights.Length;
                    for (int x = 0; x < w; x++)
                    {
                        float pdJi = pdJ[x];
                        pm[x] = pm[x] * beta1 + (1 - beta1) * pdJi;
                        pv[x] = pv[x] * beta2 + (1 - beta2) * pdJi * pdJi;
                        pw[x] -= alphat * pm[x] / ((float)Math.Sqrt(pv[x]) + epsilon);
                    }
                }

                // Biases
                fixed (float* pb = layer.Biases)
                {
                    float*
                        pdJ = dJdb,
                        pm = mB[i],
                        pv = vB[i];
                    int w = layer.Biases.Length;
                    for (int b = 0; b < w; b++)
                    {
                        float pdJi = pdJ[b];
                        pm[b] = pm[b] * beta1 + (1 - beta1) * pdJi;
                        pv[b] = pv[b] * beta2 + (1 - beta2) * pdJi * pdJi;
                        pb[b] -= alphat * pm[b] / ((float)Math.Sqrt(pv[b]) + epsilon);
                    }
                }
            }

            TrainingSessionResult result = Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationDataset, testDataset, token);

            // Cleanup
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                mW[i].Free();
                vW[i].Free();
                mB[i].Free();
                vB[i].Free();
            }
            beta1t.Free();
            beta2t.Free();
            return result;
        }

        // AdaMax method
        [NotNull]
        private static unsafe TrainingSessionResult AdaMax(
            SequentialNetwork network,
            BatchesCollection miniBatches,
            int epochs, float dropout, float eta, float beta1, float beta2,
            [CanBeNull] IProgress<BatchProgress> batchProgress,
            [CanBeNull] IProgress<TrainingProgressEventArgs> trainingProgress,
            [CanBeNull] ValidationDataset validationDataset,
            [CanBeNull] TestDataset testDataset,
            CancellationToken token)
        {
            // Initialize Adadelta parameters
            Tensor*
                mW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                uW = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                mB = stackalloc Tensor[network.WeightedLayersIndexes.Length],
                uB = stackalloc Tensor[network.WeightedLayersIndexes.Length];
            Tensor.New(1, network.WeightedLayersIndexes.Length, out Tensor beta1t);
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                WeightedLayerBase layer = network._Layers[network.WeightedLayersIndexes[i]].To<NetworkLayerBase, WeightedLayerBase>();
                Tensor.NewZeroed(1, layer.Weights.Length, out mW[i]);
                Tensor.NewZeroed(1, layer.Weights.Length, out uW[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out mB[i]);
                Tensor.NewZeroed(1, layer.Biases.Length, out uB[i]);
                beta1t[i] = beta1;
            }

            // Adadelta update for weights and biases
            void Minimize(int i, in Tensor dJdw, in Tensor dJdb, int samples, WeightedLayerBase layer)
            {
                // Alpha at timestep t
                float b1t = beta1t[i];
                beta1t[i] *= beta1;

                // Weights
                fixed (float* pw = layer.Weights)
                {
                    float*
                        pdJ = dJdw,
                        pm = mW[i],
                        pu = uW[i];
                    int w = layer.Weights.Length;
                    for (int x = 0; x < w; x++)
                    {
                        float pdJi = pdJ[x];
                        pm[x] = beta1 * pm[x] + (1 - beta1) * pdJi;
                        pu[x] = (beta2 * pu[x]).Max(pdJi.Abs());
                        pw[x] -= eta / (1 - b1t) * pm[x] / pu[x];
                    }
                }

                // Biases
                fixed (float* pb = layer.Biases)
                {
                    float*
                        pdJ = dJdb,
                        pm = mB[i],
                        pu = uB[i];
                    int w = layer.Biases.Length;
                    for (int b = 0; b < w; b++)
                    {
                        float pdJi = pdJ[b];
                        pm[b] = beta1 * pm[b] + (1 - beta1) * pdJi;
                        pu[b] = (beta2 * pu[b]).Max(pdJi.Abs());
                        pb[b] -= eta / (1 - b1t) * pm[b] / pu[b];
                    }
                }
            }

            TrainingSessionResult result = Optimize(network, miniBatches, epochs, dropout, Minimize, batchProgress, trainingProgress, validationDataset, testDataset, token);

            // Cleanup
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                mW[i].Free();
                uW[i].Free();
                mB[i].Free();
                uB[i].Free();
            }
            beta1t.Free();
            return result;
        }
    }
}
