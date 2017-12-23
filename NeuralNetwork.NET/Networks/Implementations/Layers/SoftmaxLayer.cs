using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Implementations.Layers
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
    }
}
