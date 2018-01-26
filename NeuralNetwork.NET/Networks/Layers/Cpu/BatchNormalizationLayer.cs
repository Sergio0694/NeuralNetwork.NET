using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Initialization;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// A batch normalization layer, used to improve the convergence speed of a neural network
    /// </summary>
    internal sealed class BatchNormalizationLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.BatchNormalization;

        public BatchNormalizationLayer(in TensorInfo shape, ActivationType activation) 
            : base(shape, shape, 
                WeightsProvider.NewBatchNormalizationWeights(shape), 
                WeightsProvider.NewBiases(shape.Size, BiasInitializationMode.Zero), activation) { }

        public BatchNormalizationLayer(in TensorInfo shape, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(shape, shape, w, b, activation) { }

        #region Implementation

        /// <inheritdoc/>
        public override unsafe void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            // Prepare the mu and sigma2 tensors
            Tensor.New(1, OutputInfo.Size, out Tensor mu);
            Tensor.Like(mu, out Tensor sigma2);
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pmu = mu, psigma2 = sigma2;
            Parallel.For(0, mu.Length, i =>
            {
                // Mean
                float mi = 0;
                for (int j = 0; j < n; j++)
                    mi += px[j * l + i];
                mi /= n;
                pmu[i] = mi;

                // Variance
                float sl = 0;
                for (int j = 0; j < n; j++)
                {
                    float hm = px[j * l + i] - mi;
                    sl += hm * hm;
                }
                psigma2[i] = sl / n;

            }).AssertCompleted();

            // Apply the batch normalization pass
            Tensor.Like(x, out z);
            fixed (float* pw0 = Weights, pb0 = Biases)
            {
                float* pz = z, pw = pw0, pb = pb0; // Pointers for closure
                Parallel.For(0, n, i =>
                {
                    int offset = i * l;
                    for (int j = 0; j < l; j++)
                    {
                        float hat = (x[offset + j] - pmu[j]) / (float)Math.Sqrt(psigma2[j] + float.Epsilon);
                        pz[offset + j] = pw[j] * hat + pb[j];
                    }
                }).AssertCompleted();
            }

            // Activation
            if (ActivationType == ActivationType.Identity) z.Duplicate(out a);
            else
            {
                Tensor.Like(z, out a);
                CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            throw new NotImplementedException();
        }

        #endregion

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new BatchNormalizationLayer(InputInfo, OutputInfo, Weights.AsSpan().Copy(), Biases.AsSpan().Copy());
    }
}
