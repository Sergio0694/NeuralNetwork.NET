using System.IO;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal class PoolingLayer : ConstantLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Pooling;

        [JsonProperty(nameof(OperationInfo), Order = 5)]
        private readonly PoolingInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the pooling operation performed by the layer
        /// </summary>
        public ref readonly PoolingInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        public PoolingLayer(in TensorInfo input, in PoolingInfo operation, ActivationType activation)
            : base(input, operation.GetForwardOutputTensorInfo(input), activation)
            => _OperationInfo = operation;

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            Tensor.New(x.Entities, OutputInfo.Size, out z);
            CpuDnn.PoolingForward(x, InputInfo, z);
            Tensor.New(z.Entities, z.Length, out a);
            CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx)
        {
            Tensor.Like(dy, out Tensor dy_copy);
            CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dy_copy);
            CpuDnn.PoolingBackward(x, InputInfo, dy_copy, dx);
            dy_copy.Free();
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputInfo, OperationInfo, ActivationType);

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(OperationInfo);
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="PoolingLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo _)) return null;
            if (!stream.TryRead(out ActivationType activation)) return null;
            if (!stream.TryRead(out PoolingInfo operation) && operation.Equals(PoolingInfo.Default)) return null;
            return new PoolingLayer(input, operation, activation);
        }
    }
}
