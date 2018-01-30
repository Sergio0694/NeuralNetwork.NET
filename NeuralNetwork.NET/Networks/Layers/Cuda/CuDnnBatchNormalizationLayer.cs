using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// A cuDNN-powered batch normalization layer
    /// </summary>
    internal sealed class CuDnnBatchNormalizationLayer : BatchNormalizationLayerBase
    {
        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        public CuDnnBatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, ActivationType activation) 
            : base(shape, mode, activation) { }

        public CuDnnBatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(shape, mode, w, b, activation) { }

        #region Implementation

        public override void ForwardInference(in Tensor x, out Tensor z, out Tensor a)
        {
            throw new NotImplementedException();
        }

        public override void ForwardTraining(float factor, in Tensor x, out Tensor z, out Tensor a)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                y_gpu = DnnInstance.Gpu.AllocateDevice(y),
                dy_gpu = DnnInstance.Gpu.AllocateDevice(dy),
                dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(dx.Size),
                gamma = DnnInstance.Gpu.AllocateDevice(Weights),
                beta = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length),
                mu = DnnInstance.Gpu.AllocateDevice(Mu),
                sigma2 = DnnInstance.Gpu.AllocateDevice(Sigma2))
            {
                // Backpropagation
                DnnInstance.ActivationBackward(x.Entities, x.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                DnnInstance.BatchNormalizationBackwardData(x.Entities, x.Length, x_gpu.Ptr, mu.Ptr, sigma2.Ptr, gamma.Ptr, dy_gpu.Ptr, dx_gpu.Ptr);
                dx_gpu.CopyTo(dx);

                // Gradient
                DnnInstance.BatchNormalizationBackwardGamma(x.Entities, x.Length, x_gpu.Ptr, mu.Ptr, sigma2.Ptr, dy_gpu.Ptr, gamma.Ptr);
                gamma.CopyToHost(1, Weights.Length, out dJdw);
                DnnInstance.FullyConnectedBackwardBias(dy.Entities, dy.Length, dy_gpu.Ptr, beta.Ptr);
                beta.CopyToHost(1, Biases.Length, out dJdb);
            }
        }

        #endregion

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnBatchNormalizationLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output) || input != output) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out NormalizationMode mode)) return null;
            return new CuDnnBatchNormalizationLayer(input, mode, weights, biases, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnBatchNormalizationLayer(InputInfo, NormalizationMode, Weights.AsSpan().Copy(), Biases.AsSpan().Copy(), ActivationType);
    }
}
