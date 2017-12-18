using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal class PoolingLayer : NetworkLayerBase, INetworkLayer3D
    {
        #region Parameters

        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Pooling;

        /// <inheritdoc/>
        public override int Inputs => InputInfo.Size;

        /// <inheritdoc/>
        public override int Outputs => OutputInfo.Size;

        /// <inheritdoc/>
        [JsonProperty(nameof(InputInfo), Order = 4)]
        public TensorInfo InputInfo { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputInfo), Order = 7)]
        public TensorInfo OutputInfo { get; }

        #endregion

        public PoolingLayer(TensorInfo input, ActivationFunctionType activation) : base(activation)
        {
            InputInfo = input;
            int
                outHeight = input.Height / 2 + (input.Height % 2 == 0 ? 0 : 1),
                outWidth = input.Width / 2 + (input.Width % 2 == 0 ? 0 : 1);
            OutputInfo = new TensorInfo(outHeight, outWidth, input.Channels);
        }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            x.Pool2x2(InputInfo.Channels, out z);
            z.Activation(ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime) => z.UpscalePool2x2(delta_1, InputInfo.Channels);

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputInfo, ActivationFunctionType);
    }
}
