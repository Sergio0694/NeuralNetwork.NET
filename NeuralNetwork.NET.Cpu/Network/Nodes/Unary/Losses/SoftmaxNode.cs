using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.cpuDNN;

namespace NeuralNetworkDotNet.Network.Nodes.Unary.Losses
{
    /// <summary>
    /// A custom <see cref="ActivationNode"/> used as output in a graph, which applies the softmax activation to its inputs
    /// </summary>
    internal sealed class SoftmaxNode : OutputNode
    {
        public SoftmaxNode([NotNull] Node input, Shape output)
            : base(input, output, ActivationType.Softmax, CostFunctionType.LogLikelyhood)
        { }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.Like(x);
            CpuDnn.SoftmaxForward(x, y);

            return y;
        }
    }
}
