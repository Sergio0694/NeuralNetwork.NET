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
    /// An output layer with the softmax activation function and log-likelyhood cost function
    /// </summary>
    internal class SoftmaxLayer : OutputLayerBase
    {
        /// <inheritdoc/>
        public override LayerType LayerType { get; } = LayerType.Softmax;

        public SoftmaxLayer(in TensorInfo input, int outputs, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, outputs, ActivationFunctionType.Softmax, CostFunctionType.LogLikelyhood, weightsMode, biasMode) { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            base.Forward(x, out z, out a);
            a.InPlaceSoftmaxNormalization();
        }

        public SoftmaxLayer(in TensorInfo input, int outputs, [NotNull] float[] weights, [NotNull] float[] biases)
            : base(input, outputs, weights, biases, ActivationFunctionType.Softmax, CostFunctionType.LogLikelyhood) { }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new SoftmaxLayer(InputInfo, OutputInfo.Size, Weights.BlockCopy(), Biases.BlockCopy());

        /// <summary>
        /// Tries to deserialize a new <see cref="SoftmaxLayer"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the layer data</param>
        [MustUseReturnValue, CanBeNull]
        public new static INetworkLayer Deserialize([NotNull] Stream stream)
        {
            if (!stream.TryRead(out TensorInfo input)) return null;
            if (!stream.TryRead(out TensorInfo output)) return null;
            if (!stream.TryRead(out ActivationFunctionType activation) && activation == ActivationFunctionType.Softmax) return null;
            if (!stream.TryRead(out int wLength)) return null;
            float[] weights = stream.ReadUnshuffled(wLength);
            if (!stream.TryRead(out int bLength)) return null;
            float[] biases = stream.ReadUnshuffled(bLength);
            if (!stream.TryRead(out CostFunctionType cost) && cost == CostFunctionType.LogLikelyhood) return null;
            return new SoftmaxLayer(input, output.Size, weights, biases);
        }
    }
}
