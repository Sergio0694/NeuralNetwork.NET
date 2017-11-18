using System;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A pooling layer, with a 2x2 window and a stride of 2
    /// </summary>
    internal class PoolingLayer : NetworkLayerBase
    {
        /// <inheritdoc/>
        public override int Inputs { get; }

        /// <inheritdoc/>
        public override int Outputs { get; }

        public PoolingLayer(int height, int width, int depth) : base(ActivationFunctionType.Identity)
        {
            if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException("The height and width must be positive numbers");
            if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth), "The depth must be at least equal to 1");
            Inputs = height * width * depth;
            Outputs = (height / 2 + (height % 2 == 0 ? 0 : 1)) * (width / 2 + (width % 2 == 0 ? 0 : 1)) * depth;
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
