using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using System;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A fully connected (dense) network layer
    /// </summary>
    internal class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.FullyConnected;

        public FullyConnectedLayer(in TensorInfo input, int neurons, ActivationFunctionType activation, BiasInitializationMode biasMode)
            : base(input, TensorInfo.CreateLinear(neurons),
                  WeightsProvider.NewFullyConnectedWeights(input.Size, neurons),
                  WeightsProvider.NewBiases(neurons, biasMode), activation) { }

        public FullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(TensorInfo.CreateLinear(weights.GetLength(0)), TensorInfo.CreateLinear(weights.GetLength(1)), weights, biases, activation)
        {
            if (weights.GetLength(1) != biases.Length)
                throw new ArgumentException("The biases vector must have the same size as the number of output neurons");
        }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            x.MultiplyWithSum(Weights, Biases, out z);
            z.Activation(ActivationFunctions.Activation, out a);
        }

        /// <inheritdoc/>
        public override void Backpropagate(in Tensor delta_1, in Tensor z, ActivationFunction activationPrime)
        {
            Weights.Transpose(out Tensor wt);
            z.InPlaceMultiplyAndHadamardProductWithActivationPrime(delta_1, wt, activationPrime);
            wt.Free();
        }

        /// <inheritdoc/>
        public override void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb)
        {
            a.Transpose(out Tensor at);
            at.Multiply(delta, out dJdw);
            at.Free();
            delta.CompressVertically(out dJdb);
        }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new FullyConnectedLayer(Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType);
    }
}
