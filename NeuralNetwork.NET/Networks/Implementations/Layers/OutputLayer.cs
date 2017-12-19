using System;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Misc;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// An output layer with a variable cost function
    /// </summary>
    internal sealed class OutputLayer : OutputLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Output;

        public OutputLayer(in TensorInfo input, int neurons, ActivationFunctionType activation, CostFunctionType cost, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, neurons, activation, cost, weightsMode, biasMode)
        {
            if (activation == ActivationFunctionType.Softmax || cost == CostFunctionType.LogLikelyhood)
                throw new ArgumentException("The softmax activation and log-likelyhood cost function must be used together in a softmax layer");
            if (activation != ActivationFunctionType.Sigmoid && cost == CostFunctionType.CrossEntropy)
                throw new ArgumentException("The cross-entropy cost function can only accept inputs in the (0,1) range");
        }

        public OutputLayer([NotNull] float[,] weights, [NotNull] float[] biases, ActivationFunctionType activation, CostFunctionType cost)
            : base(weights, biases, activation, cost) { }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new OutputLayer(Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType, CostFunctionType);
    }
}
