using System;
using System.IO;
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
    internal sealed class BatchNormalizationLayer : WeightedLayerBase, IDisposable
    {
        // Cached mu tensor
        private Tensor _Mu;

        // Cached sigma^2 tensor
        private Tensor _Sigma2;

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
            _Mu.TryFree();
            Tensor.New(1, OutputInfo.Size, out _Mu);
            _Sigma2.TryFree();
            Tensor.Like(_Mu, out _Sigma2);
            int
                n = x.Entities,
                l = x.Length;
            float* px = x, pmu = _Mu, psigma2 = _Sigma2;
            Parallel.For(0, _Mu.Length, i =>
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
                        float hat = (px[offset + j] - pmu[j]) / (float)Math.Sqrt(psigma2[j] + float.Epsilon);
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
        public override unsafe void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            // Activation backward
            int
                n = x.Entities,
                l = x.Length;
            Tensor.Like(dy, out Tensor dy_copy);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dy_copy);
            
            // Gamma gradient
            Tensor.New(1, l, out dJdw);
            float* px = x, pdy = dy_copy, pdJdw = dJdw, pmu = _Mu, psigma2 = _Sigma2;
            Parallel.For(0, l, j =>
            {
                float sum = 0;
                for (int i = 0; i < n; i++)
                {
                    float hat = (px[i * l + i] - pmu[j]) / (float)Math.Sqrt(psigma2[j] + float.Epsilon);
                    sum += pdy[i * l + j] * hat;
                }
                pdJdw[j] = sum;
            }).AssertCompleted();

            // Beta gradient
            Tensor.New(1, l, out dJdb);
            CpuDnn.FullyConnectedBackwardBias(dy_copy, dJdb); // Same as fully connected, vertical sum

            // Input error delta
            fixed (float* pw0 = Weights)
            {
                float* pdx = dx, pw = pw0;
                Parallel.For(0, n, i =>
                {
                    for (int j = 0; j < l; j++)
                    {
                        float
                            left = 1f / n * pw[j] / (float)Math.Sqrt(psigma2[j] + float.Epsilon),
                            _1st = n * pdy[i * l + j],
                            _2nd = 0,
                            _3rdLeft = (px[i * l + j] - pmu[j]) * 1f / (psigma2[j] + float.Epsilon),
                            _3rdRight = 0;
                        for (int k = 0; k < n; k++)
                        {
                            float pdykj = pdy[k * l + j];
                            _2nd += pdykj;
                            _3rdRight += pdykj * (px[k * l + j] - pmu[j]);
                        }
                        pdx[i * l + j] = left * _1st - _2nd - _3rdLeft * _3rdRight;
                    }
                }).AssertCompleted();
            }
            dy_copy.Free();
        }

        #endregion

        /// <summary>
        /// Tries to deserialize a new <see cref="BatchNormalizationLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output) || input != output) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            return new BatchNormalizationLayer(input, weights, biases, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new BatchNormalizationLayer(InputInfo, Weights.AsSpan().Copy(), Biases.AsSpan().Copy(), ActivationType);

        #region IDisposable

        ~BatchNormalizationLayer() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose() => Dispose();

        // Disposes the temporary tensors
        private void Dispose()
        {
            _Mu.TryFree();
            _Sigma2.TryFree();
        }

        #endregion
    }
}
