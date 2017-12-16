using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.Helpers;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// A fully connected (dense) network layer
    /// </summary>
    internal class FullyConnectedLayer : WeightedLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.FullyConnected;

        /// <inheritdoc/>
        public override int Inputs => Weights.GetLength(0);

        /// <inheritdoc/>
        public override int Outputs => Weights.GetLength(1);

        public FullyConnectedLayer(int inputs, int outputs, ActivationFunctionType activation)
            : base(WeightsProvider.FullyConnectedWeights(inputs, outputs),
                WeightsProvider.Biases(outputs), activation)
        { }

        public FullyConnectedLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation)
            : base(weights, biases, activation) { }

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
