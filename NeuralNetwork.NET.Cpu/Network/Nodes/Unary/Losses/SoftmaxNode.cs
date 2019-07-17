using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Nodes.Enums;

namespace NeuralNetworkDotNet.Network.Nodes.Unary.Losses
{
    /// <summary>
    /// A custom <see cref="ActivationNode"/> used as output in a graph, which applies the softmax activation to its inputs
    /// </summary>
    internal sealed class SoftmaxNode : OutputNode
    {
        /// <inheritdoc/>
        public override NodeType Type => NodeType.Softmax;

        public SoftmaxNode([NotNull] Node input) : base(input, ActivationType.Softmax, CostFunctionType.LogLikelyhood)
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
