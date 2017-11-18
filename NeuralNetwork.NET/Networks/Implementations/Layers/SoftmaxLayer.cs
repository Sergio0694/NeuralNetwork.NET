using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;

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
        public override (float[,] Z, float[,] A) Forward(float[,] x)
        {
            (float[,] z, float[,] a) = base.Forward(x);
            a.InPlaceSoftmaxNormalization();
            return (z, a);
        }
    }
}
