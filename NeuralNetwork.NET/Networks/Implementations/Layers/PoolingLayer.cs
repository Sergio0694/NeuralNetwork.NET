using NeuralNetworkNET.Convolution.Misc;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
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
        public override int Inputs => InputVolume.Size;

        /// <inheritdoc/>
        public override int Outputs => OutputVolume.Size;

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
            int outAxis = input.Axis / 2 + (input.Axis % 2 == 0 ? 0 : 1);
            OutputVolume = new VolumeInformation(outAxis, input.Depth);
        }

        /// <inheritdoc/>
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            float[,]
                z = x.Pool2x2(InputVolume.Depth),
                a = ActivationFunctionType == ActivationFunctionType.Identity
                    ? z.BlockCopy()
                    : z.Activation(ActivationFunctions.Activation);
            return (z, a);
        }

        /// <inheritdoc/>
        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction _)
        {
            return z.UpscalePool2x2(delta_1, InputVolume.Depth);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new PoolingLayer(InputVolume, ActivationFunctionType);
    }
}
