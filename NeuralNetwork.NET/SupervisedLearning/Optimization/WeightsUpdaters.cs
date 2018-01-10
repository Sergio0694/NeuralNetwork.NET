using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Algorithms.Info;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    internal static class WeightsUpdaters
    {
        // Classic SGD algorithm
        [Pure, NotNull]
        public static WeightsUpdater StochasticGradientDescent([NotNull] StochasticGradientDescentInfo info)
        {
            float 
                eta = info.Eta,
                lambda = info.Lambda;
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

            return Minimize;
        }

        // Adadelta method
        [Pure, NotNull]
        public static WeightsUpdater Adadelta([NotNull] AdadeltaInfo info, [NotNull] SequentialNetwork network)
        {
            // Initialize Adadelta parameters
            float
                rho = info.Rho,
                epsilon = info.Epsilon,
                l2 = info.L2;
            float[][]
                egSquaredW = new float[network.WeightedLayersIndexes.Length][],
                eDeltaxSquaredW = new float[network.WeightedLayersIndexes.Length][],
                egSquaredB = new float[network.WeightedLayersIndexes.Length][],
                eDeltaxSquaredB = new float[network.WeightedLayersIndexes.Length][];
            for (int i = 0; i < network.WeightedLayersIndexes.Length; i++)
            {
                WeightedLayerBase layer = network._Layers[network.WeightedLayersIndexes[i]].To<NetworkLayerBase, WeightedLayerBase>();
                egSquaredW[i] = new float[layer.Weights.Length];
                eDeltaxSquaredW[i] = new float[layer.Weights.Length];
                egSquaredB[i] = new float[layer.Biases.Length];
                eDeltaxSquaredB[i] = new float[layer.Biases.Length];
            }

            // Adadelta update for weights and biases
            unsafe void Minimize(int i, in Tensor dJdw, in Tensor dJdb, int samples, WeightedLayerBase layer)
            {
                fixed (float* pw = layer.Weights, egSqrt = egSquaredW[i], eDSqrtx = eDeltaxSquaredW[i])
                {
                    float* pdj = dJdw;
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
                        pw[x] += dx - l2 * pw[x];
                    }
                }

                // Tweak the biases of the lth layer
                fixed (float* pb = layer.Biases, egSqrt = egSquaredB[i], eDSqrtb = eDeltaxSquaredB[i])
                {
                    float* pdj = dJdb;
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
                        pb[b] += db - l2 * pb[b];
                    }
                }
            }

            return Minimize;
        }

        // Adam method
        [Pure, NotNull]
        public static WeightsUpdater Adam([NotNull] AdamInfo info, [NotNull] SequentialNetwork network)
        {
            throw new NotImplementedException();
        }

        // AdaMax method
        [Pure, NotNull]
        public static WeightsUpdater AdaMax([NotNull] AdaMaxInfo info, [NotNull] SequentialNetwork network)
        {
            throw new NotImplementedException();
        }
    }
}
