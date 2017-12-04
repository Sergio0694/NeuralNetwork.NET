using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal sealed class PoolingLayer : NetworkLayerBase, INetworkLayer3D
    {
        #region Parameters

        /// <inheritdoc/>
        public override int Inputs => InputVolume.Volume;

        /// <inheritdoc/>
        public override int Outputs => OutputVolume.Volume;

        /// <inheritdoc/>
        [JsonProperty(nameof(InputVolume), Order = 4)]
        public VolumeInformation InputVolume { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputVolume), Order = 7)]
        public VolumeInformation OutputVolume { get; }

        #endregion

        public PoolingLayer(VolumeInformation input, ActivationFunctionType activation) : base(activation)
        {
            InputVolume = input;
            int
                outHeight = input.Height / 2 + (input.Height % 2 == 0 ? 0 : 1),
                outWidth = input.Width / 2 + (input.Width % 2 == 0 ? 0 : 1);
            OutputVolume = (outHeight, outWidth, input.Depth);
        }

        /// <inheritdoc/>
        public override void Forward(in FloatSpan2D x, out FloatSpan2D z, out FloatSpan2D a)
        {
            x.Pool2x2(InputVolume.Depth, out z);
            z.Activation(ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in FloatSpan2D delta_1, in FloatSpan2D z, ActivationFunction activationPrime) => z.UpscalePool2x2(delta_1, InputVolume.Depth);

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputVolume, ActivationFunctionType);
    }
}
