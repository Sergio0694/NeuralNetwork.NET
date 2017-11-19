using System;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    internal class PoolingLayer : NetworkLayerBase, INetworkLayer3D
    {
        #region Parameters

        /// <inheritdoc/>
        public override int Inputs => InputVolume.Size;

        /// <inheritdoc/>
        public override int Outputs => OutputVolume.Size;

        /// <inheritdoc/>
        public VolumeInformation InputVolume { get; }

        /// <inheritdoc/>
        public VolumeInformation OutputVolume { get; }

        #endregion

        public PoolingLayer(VolumeInformation input) : base(ActivationFunctionType.Identity)
        {
            int outAxis = input.Axis / 2 + (input.Axis % 2 == 0 ? 0 : 1);
            OutputVolume = new VolumeInformation(outAxis, input.Depth);
        }

        /// <inheritdoc/>
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            throw new NotImplementedException();
        }

        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }
    }
}
