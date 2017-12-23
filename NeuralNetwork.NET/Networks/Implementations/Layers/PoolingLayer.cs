using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using Newtonsoft.Json;
using System.IO;
using System.Runtime.CompilerServices;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal class PoolingLayer : NetworkLayerBase
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

        public PoolingLayer(in TensorInfo input, in PoolingInfo operation, ActivationFunctionType activation)
            : base(input, new TensorInfo(
                input.Height / 2 + (input.Height % 2 == 0 ? 0 : 1),
                input.Width / 2 + (input.Width % 2 == 0 ? 0 : 1),
                input.Channels), activation)
            => _OperationInfo = operation;

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            x.Pool2x2(InputInfo.Channels, out z);
            z.Activation(ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime) => z.UpscalePool2x2(delta_1, InputInfo.Channels);

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputInfo, OperationInfo, ActivationFunctionType);

        /// <inheritdoc/>
        public override void Serialize([NotNull] Stream stream)
        {
            base.Serialize(stream);
            stream.Write(OperationInfo);
        }
    }
}
