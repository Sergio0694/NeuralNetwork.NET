using System;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.Services;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// A pooling layer running on cuDNN, with a custom pooling mode
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal sealed class CuDnnPoolingLayer : PoolingLayer, IDisposable
    {
        #region cuDNN fields

        // The NCHW tensor info for the layer inputs
        [NotNull]
        private readonly TensorDescriptor InputDescription = new TensorDescriptor();

        // The descriptor for the pooling operation performed by the layer
        [NotNull]
        private readonly PoolingDescriptor PoolingDescription = new PoolingDescriptor();

        // The NCHW tensor info for the layer outputs
        [NotNull]
        private readonly TensorDescriptor OutputDescription = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        #endregion

        #region Fields

        // A copy of the layer output activity
        private Tensor _Z;

        #endregion

        public CuDnnPoolingLayer(in TensorInfo input, in PoolingInfo operation, ActivationFunctionType activation) : base(input, operation, activation)
        {
            PoolingDescription.Set2D((PoolingMode)operation.Mode, NanPropagation.PROPAGATE_NAN, operation.WindowHeight, operation.WindowWidth, operation.VerticalPadding, operation.HorizontalPadding, operation.VerticalStride, operation.HorizontalStride);
        }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * OutputInfo.Size))
            {
                // Pooling
                InputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputInfo.Channels, InputInfo.Height, InputInfo.Width);
                OutputDescription.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OutputInfo.Channels, OutputInfo.Height, OutputInfo.Width);
                DnnInstance.PoolingForward(PoolingDescription, 1, InputDescription, x_gpu.Ptr, 0, OutputDescription, z_gpu.Ptr);
                z_gpu.CopyToHost(x.Entities, OutputInfo.Size, out z);
                _Z.TryFree();
                z.Duplicate(out _Z);

                // Activation
                DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z_gpu.Ptr, ActivationFunctions.Activation);
                z_gpu.CopyToHost(z.Entities, z.Length, out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor dy, in Tensor z, ActivationFunction activationPrime)
        {
            using (DeviceMemory<float> dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(z.Size))
            {
                using (DeviceMemory<float>
                    x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                    y_gpu = DnnInstance.Gpu.AllocateDevice(_Z),
                    dy_gpu = DnnInstance.Gpu.AllocateDevice(dy))
                {
                    DnnInstance.PoolingBackward(PoolingDescription, 1, OutputDescription, y_gpu.Ptr, OutputDescription, dy_gpu.Ptr, InputDescription, x_gpu.Ptr, 0, InputDescription, dx_gpu.Ptr);
                }
                using (DeviceMemory<float> z_gpu = DnnInstance.Gpu.AllocateDevice(z))
                {
                    DnnInstance.ActivationBackward(z.Entities, z.Length, z_gpu.Ptr, dx_gpu.Ptr, activationPrime);
                    z_gpu.CopyTo(z);
                }
            }
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnPoolingLayer(InputInfo, OperationInfo, ActivationFunctionType);

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnPoolingLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo _)) return null;
            if (!stream.TryRead(out ActivationFunctionType activation)) return null;
            if (!stream.TryRead(out PoolingInfo operation)) return null;
            return new CuDnnPoolingLayer(input, operation, activation);
        }

        #region IDisposable

        ~CuDnnPoolingLayer() => Dispose();

        /// <inheritdoc/>
        void IDisposable.Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose();
        }

        // Private Dispose method
        private void Dispose()
        {
            _Z.TryFree();
        }

        #endregion
    }
}