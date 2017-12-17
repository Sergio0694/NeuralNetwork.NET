using Alea;
using Alea.cuDNN;
using NeuralNetworkNET.Cuda.Extensions;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Cuda.Services;

namespace NeuralNetworkNET.Cuda.Layers
{
    /// <summary>
    /// A pooling layer running on cuDNN, with a 2x2 window and a stride of 2
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal sealed class CuDnnPoolingLayer : PoolingLayer
    {
        #region cuDNN fields

        // The NCHW tensor info for the layer inputs
        [NotNull]
        private readonly TensorDescriptor InputInfo = new TensorDescriptor();

        // The descriptor for the pooling operation performed by the layer
        [NotNull]
        private readonly PoolingDescriptor PoolDescriptor = new PoolingDescriptor();

        // The NCHW tensor info for the layer outputs
        [NotNull]
        private readonly TensorDescriptor OutputInfo = new TensorDescriptor();

        /// <summary>
        /// Gets the <see cref="Dnn"/> instance for the current layer
        /// </summary>
        [NotNull]
        private readonly Dnn DnnInstance = DnnService.Instance;

        #endregion

        public CuDnnPoolingLayer(VolumeInformation input, ActivationFunctionType activation) : base(input, activation)
        {
            PoolDescriptor.Set2D(PoolingMode.MAX, NanPropagation.PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);
        }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            using (DeviceMemory<float>
                x_gpu = DnnInstance.Gpu.AllocateDevice(x),
                z_gpu = DnnInstance.Gpu.AllocateDevice<float>(x.Entities * Outputs))
            {
                // Pooling
                InputInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, InputVolume.Depth, InputVolume.Height, InputVolume.Width);
                OutputInfo.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, x.Entities, OutputVolume.Depth, OutputVolume.Height, OutputVolume.Width);
                DnnInstance.PoolingForward(PoolDescriptor, 1, InputInfo, x_gpu.Ptr, 0, OutputInfo, z_gpu.Ptr);
                z_gpu.CopyToHost(x.Entities, Outputs, out z);

                // Activation
                DnnInstance.ActivationForward(z.Entities, z.Length, z_gpu.Ptr, z.Length, z_gpu.Ptr, z.Length, ActivationFunctions.Activation);
                z_gpu.CopyToHost(z.Entities, z.Length, out a);
            }
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime) => z.UpscalePool2x2(delta_1, InputVolume.Depth);

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputVolume, ActivationFunctionType);
    }
}