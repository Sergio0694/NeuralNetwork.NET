using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.cpuDNN;

namespace NeuralNetworkDotNet.Network.Layers
{
    /// <summary>
    /// A custom <see cref="ActivationLayer"/> used as output in a graph, which applies the softmax activation to its inputs
    /// </summary>
    internal sealed class SoftmaxLayer : OutputLayer
    {
        public SoftmaxLayer(Shape input, Shape output)
            : base(input, output, ActivationType.Softmax, CostFunctionType.LogLikelyhood)
        { }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.Like(x);
            CpuDnn.SoftmaxForward(x, y);

            return y;
        }

        /// <inheritdoc/>
        public override ILayer Clone() => new SoftmaxLayer(InputShape, OutputShape);
    }
}
