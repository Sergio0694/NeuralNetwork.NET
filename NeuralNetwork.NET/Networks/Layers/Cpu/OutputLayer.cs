using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Layers.Cpu
{
    /// <summary>
    /// An output layer with a variable cost function
    /// </summary>
    internal sealed class OutputLayer : OutputLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Output;

        public OutputLayer(in TensorInfo input, int outputs, ActivationFunctionType activation, CostFunctionType cost, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, outputs, activation, cost, weightsMode, biasMode)
        {
            if (activation == ActivationFunctionType.Softmax || cost == CostFunctionType.LogLikelyhood)
                throw new ArgumentException("The softmax activation and log-likelyhood cost function must be used together in a softmax layer");
            if (activation != ActivationFunctionType.Sigmoid && cost == CostFunctionType.CrossEntropy)
                throw new ArgumentException("The cross-entropy cost function can only accept inputs in the (0,1) range");
        }

        public OutputLayer(in TensorInfo input, int outputs, [NotNull] float[] weights, [NotNull] float[] biases, ActivationFunctionType activation, CostFunctionType cost)
            : base(input, outputs, weights, biases, activation, cost) { }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new OutputLayer(InputInfo, OutputInfo.Size, Weights.BlockCopy(), Biases.BlockCopy(), ActivationFunctionType, CostFunctionType);

        /// <summary>
        /// Tries to deserialize a new <see cref="OutputLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationFunctionType activation)) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out CostFunctionType cost)) return null;
            return new OutputLayer(input, output.Size, weights, biases, activation, cost);
        }
    }
}
