using System;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Networks.Implementations.Misc;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A convolutional layer, used in a CNN network
    /// </summary>
    internal class ConvolutionalLayer : WeightedLayerBase
    {
        public override int Inputs { get; }
        public override int Outputs { get; }

        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            throw new NotImplementedException();
        }

        public override float[,] Backpropagate(float[,] delta_1, float[,] z, ActivationFunction activationPrime)
        {
            throw new NotImplementedException();
        }

        public ConvolutionalLayer(int height, int width, int depth, int kernels, ActivationFunctionType activation)
            : base(WeightsProvider.ConvolutionalKernels(height, width, depth, kernels),
                WeightsProvider.Biases(kernels), activation)
        { }

        public override LayerGradient ComputeGradient(float[,] a, float[,] delta)
        {
            throw new NotImplementedException();
        }
    }
}
