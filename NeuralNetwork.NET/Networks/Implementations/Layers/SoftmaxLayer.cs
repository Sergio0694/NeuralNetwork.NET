using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.Networks.Implementations.Layers
{
    /// <summary>
    /// An output layer with the softmax activation function and log-likelyhood cost function
    /// </summary>
    internal sealed class SoftmaxLayer : OutputLayerBase
    {
        public SoftmaxLayer(int inputs, int outputs)
            : base(inputs, outputs, ActivationFunctionType.Softmax, CostFunctionType.LogLikelyhood)
        { }

        /// <inheritdoc/>
        public override void Forward(in Tensor x, out Tensor z, out Tensor a)
        {
            base.Forward(x, out z, out a);
            a.InPlaceSoftmaxNormalization();
        }

        public SoftmaxLayer([NotNull] float[,] weights, [NotNull] float[] biases)
            : base(weights, biases, ActivationFunctionType.Softmax, CostFunctionType.LogLikelyhood) { }

        /// <inheritdoc/>
        public override INetworkLayer Clone() => new SoftmaxLayer(Weights.BlockCopy(), Biases.BlockCopy());
    }
}
