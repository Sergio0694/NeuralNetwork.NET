using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
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
        // The NCHW tensor info for the layer inputs and outputs
        [NotNull]
        private readonly TensorDescriptor DataDescription = new TensorDescriptor();

        // The NCHW tensor info for the batch normalization parameters
        [NotNull]
        private readonly TensorDescriptor BatchNormalizationDescription = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        // cuDNN fields setup
        private void SetupCuDnnInfo()
        {
            BatchNormalizationDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, 1, Mu.Size, 1, 1);
        }

        public CuDnnBatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, ActivationType activation)
            : base(shape, mode, activation)
            => SetupCuDnnInfo();

        public CuDnnBatchNormalizationLayer(in TensorInfo shape, NormalizationMode mode, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(shape, mode, w, b, activation)
            => SetupCuDnnInfo();

        #region Implementation

        /// <inheritdoc/>
        public override void ForwardInference(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                gamma_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                beta_gpu = DnnInstance.Gpu.AllocateDevice(Biases),
                mu_gpu = DnnInstance.Gpu.AllocateDevice(Mu),
                sigma2_gpu = DnnInstance.Gpu.AllocateDevice(Sigma2),
                y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size))
            {
                if (NormalizationMode == NormalizationMode.PerActivation) DataDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, x.Length, 1, 1);
                DataDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                DnnInstance.BatchNormalizationForwardInference(
                    (BatchNormMode)NormalizationMode, 1, 0, DataDescription, x_gpu.Ptr, DataDescription, y_gpu.Ptr, 
                    BatchNormalizationDescription, gamma_gpu.Ptr, beta_gpu.Ptr,
                    mu_gpu.Ptr, sigma2_gpu.Ptr, CpuDnn.CUDNN_BN_MIN_EPSILON);
                y_gpu.CopyToHost(x.Entities, x.Length, out z);
                DnnInstance.ActivationForward(x.Entities, x.Length, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                y_gpu.CopyToHost(x.Entities, x.Length, out a);
            }
        }

        /// <inheritdoc/>
        public override void ForwardTraining(float factor, in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                gamma_gpu = DnnInstance.Gpu.AllocateDevice(Weights),
                beta_gpu = DnnInstance.Gpu.AllocateDevice(Biases),
                mu_gpu = DnnInstance.Gpu.AllocateDevice(Mu),
                sigma2_gpu = DnnInstance.Gpu.AllocateDevice(Sigma2),
                y_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size))
            {
                if (NormalizationMode == NormalizationMode.PerActivation) DataDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, x.Length, 1, 1);
                DataDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                DnnInstance.BatchNormalizationForwardTraining(
                    (BatchNormMode)NormalizationMode, 1, 0, DataDescription, x_gpu.Ptr, DataDescription, y_gpu.Ptr,
                    BatchNormalizationDescription, gamma_gpu.Ptr, beta_gpu.Ptr, factor, mu_gpu.Ptr, sigma2_gpu.Ptr, CpuDnn.CUDNN_BN_MIN_EPSILON,
                    default, default); // TODO: store cache
                y_gpu.CopyToHost(x.Entities, x.Length, out z);
                DnnInstance.ActivationForward(x.Entities, x.Length, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                y_gpu.CopyToHost(x.Entities, x.Length, out a);
            }
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
                dgamma = DnnInstance.Gpu.AllocateDevice<float>(Weights.Length),
                dbeta = DnnInstance.Gpu.AllocateDevice<float>(Biases.Length))
            {
                // Backpropagation
                DnnInstance.ActivationBackward(x.Entities, x.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                DnnInstance.BatchNormalizationBackward(
                    (BatchNormMode)NormalizationMode, 1, 0, 1, 0,
                    DataDescription, x_gpu.Ptr, DataDescription, dy_gpu.Ptr, DataDescription, dx_gpu.Ptr,
                    BatchNormalizationDescription, gamma.Ptr, dgamma.Ptr, dbeta.Ptr,
                    CpuDnn.CUDNN_BN_MIN_EPSILON, default, default); // TODO: use cache
                dx_gpu.CopyTo(dx);
                dgamma.CopyToHost(1, Weights.Length, out dJdw);
                dbeta.CopyToHost(1, Biases.Length, out dJdb);
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
