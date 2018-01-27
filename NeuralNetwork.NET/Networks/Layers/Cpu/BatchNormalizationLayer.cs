using System;
using System.IO;
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
            if (_Mu.IsNull) Tensor.New(1, OutputInfo.Size, out _Mu);
            if (_Sigma2.IsNull) Tensor.Like(_Mu, out _Sigma2);
            Tensor.Like(x, out z);
            fixed (float* pw = Weights, pb = Biases)
            {
                Tensor.Reshape(pw, 1, Weights.Length, out Tensor w);
                Tensor.Reshape(pb, 1, Biases.Length, out Tensor b);
                CpuDnn.BatchNormalizationForward(x, _Mu, _Sigma2, w, b, z);
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
            Tensor.Like(dy, out Tensor dy_copy);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dy_copy);
            
            // Gamma gradient
            Tensor.New(1, Weights.Length, out dJdw);
            CpuDnn.BatchNormalizationBackwardGamma(x, _Mu, _Sigma2, dy, dJdw);

            // Beta gradient
            Tensor.New(1, Biases.Length, out dJdb);
            CpuDnn.FullyConnectedBackwardBias(dy_copy, dJdb); // Same as fully connected, vertical sum

            // Input error delta
            fixed (float* pw = Weights)
            {
                Tensor.Reshape(pw, 1, Weights.Length, out Tensor w);
                CpuDnn.BatchNormalizationBackwardData(x, _Mu, _Sigma2, w, dy, dx);
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
