using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Cpu;

namespace NeuralNetworkNET.Networks.Layers.Cuda
{
    /// <summary>
    /// A pooling layer running on cuDNN, with a custom pooling mode
    /// </summary>
    internal sealed class CuDnnPoolingLayer : PoolingLayer
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
        private readonly Dnn DnnInstance = CuDnnService.Instance;

        #endregion

        public CuDnnPoolingLayer(in TensorInfo input, in PoolingInfo operation, ActivationType activation) : base(input, operation, activation)
        {
            PoolingDescription.Set2D((Alea.cuDNN.PoolingMode)operation.Mode, NanPropagation.PROPAGATE_NAN, operation.WindowHeight, operation.WindowWidth, operation.VerticalPadding, operation.HorizontalPadding, operation.VerticalStride, operation.HorizontalStride);
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

                // Activation
                DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z_gpu.Ptr, ActivationFunctions.Activation);
                z_gpu.CopyToHost(z.Entities, z.Length, out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                dx_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Size),
                y_gpu = DnnInstance.Gpu.AllocateDevice(y),
                dy_gpu = DnnInstance.Gpu.AllocateDevice(dy))
            {
                DnnInstance.ActivationBackward(y.Entities, y.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                DnnInstance.PoolingBackward(PoolingDescription, 1, OutputDescription, y_gpu.Ptr, OutputDescription, dy_gpu.Ptr, InputDescription, x_gpu.Ptr, 0, InputDescription, dx_gpu.Ptr);
                dx_gpu.CopyTo(dx);
            }
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="CuDnnPoolingLayer"/> from the input <see cref="System.IO.Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="System.IO.Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] System.IO.Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo _)) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out PoolingInfo operation)) return null;
            return new CuDnnPoolingLayer(input, operation, activation);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new CuDnnPoolingLayer(InputInfo, OperationInfo, ActivationType);
    }
}