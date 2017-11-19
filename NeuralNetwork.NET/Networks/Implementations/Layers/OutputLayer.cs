using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// An output layer with a variable cost function
    /// </summary>
    internal sealed class OutputLayer : OutputLayerBase
    {
        public OutputLayer(int inputs, int outputs, ActivationFunctionType activation, CostFunctionType cost)
            : base(inputs, outputs, activation, cost)
        {
            if (activation == ActivationFunctionType.Softmax || cost == CostFunctionType.LogLikelyhood)
                throw new ArgumentException("The softmax activation and log-likelyhood cost function must be used together in a softmax layer");
        }

        private OutputLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation, CostFunctionType cost)
            : base(weights, biases, activation, cost) { }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new OutputLayer(Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType, CostFunctionType);
    }
}
