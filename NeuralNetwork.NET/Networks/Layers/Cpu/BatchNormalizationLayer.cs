using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// A batch normalization layer, used to improve the convergence speed of a neural network
    /// </summary>
    internal sealed class BatchNormalizationLayer : BatchNormalizationLayerBase
    {
        public BatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, ActivationType activation)
            : base(shape, mode, activation) { }

        public BatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(shape, mode, w, b, activation) { }

        #region Implementation

        /// <inheritdoc/>
        public override unsafe void ForwardInference(in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights, pb = Biases)
            {
                Tensor.Reshape(pw, 1, Mu.Length, out Tensor gamma);
                Tensor.Reshape(pb, 1, Mu.Length, out Tensor beta);
                Tensor.Like(x, out z);
                CpuDnn.BatchNormalizationForward(NormalizationMode, InputInfo, x, Mu, Sigma2, gamma, beta, z);
                Tensor.Like(z, out a);
                CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
            }
        }

        /// <inheritdoc/>
        public override unsafe void ForwardTraining(float factor, in Tensor x, out Tensor z, out Tensor a)
        {
            fixed (float* pw = Weights, pb = Biases)
            {
                Tensor.Reshape(pw, 1, Mu.Length, out Tensor gamma);
                Tensor.Reshape(pb, 1, Mu.Length, out Tensor beta);
                Tensor.Like(x, out z);
                CpuDnn.BatchNormalizationForward(NormalizationMode, InputInfo, x, factor, Mu, Sigma2, gamma, beta, z);
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

            // Input error delta
            fixed (float* pw = Weights)
            {
                Tensor.Reshape(pw, 1, Weights.Length, out Tensor w);
                CpuDnn.BatchNormalizationBackwardData(NormalizationMode, InputInfo, x, Mu, Sigma2, w, dy_copy, dx);
            }
            
            // Gamma gradient
            Tensor.New(1, Weights.Length, out dJdw);
            CpuDnn.BatchNormalizationBackwardGamma(NormalizationMode, InputInfo, x, Mu, Sigma2, dy_copy, dJdw);

            // Beta gradient
            Tensor.New(1, Biases.Length, out dJdb);
            CpuDnn.BatchNormalizationBackwardBeta(NormalizationMode, InputInfo, dy_copy, dJdb);
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
            if (!stream.TryRead(out NormalizationMode mode)) return null;
            return new BatchNormalizationLayer(input, mode, weights, biases, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new BatchNormalizationLayer(InputInfo, NormalizationMode, Weights.AsSpan().Copy(), Biases.AsSpan().Copy(), ActivationType);
    }
}
