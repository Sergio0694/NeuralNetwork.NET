using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Nodes.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Binary
{
    /// <summary>
    /// A depth concatenation node, to merge multiple input nodes together
    /// </summary>
    internal sealed class DepthConcatenationNode : BinaryNodeBase
    {
        public DepthConcatenationNode([NotNull] Node left, [NotNull] Node right)
            : base(left, right, (left.Shape.C + right.Shape.C, left.Shape.H, left.Shape.W))
        { }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x1, Tensor x2)
        {
            var y = Tensor.New(x1.Shape.N, Shape.C, Shape.H, Shape.W);
            CpuDnn.DepthConcatenationForward(x1, x2, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x1, Tensor x2, Tensor y, Tensor dy)
        {
            return dy.Clone();
        }
    }
}
