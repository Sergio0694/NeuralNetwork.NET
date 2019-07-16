using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Nodes.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Binary
{
    /// <summary>
    /// A sum node, that merges inputs from two nodes together
    /// </summary>
    internal sealed class SumNode : BinaryNodeBase
    {
        public SumNode([NotNull] Node left, [NotNull] Node right) : base(left, right, left.Shape)
        {
            Guard.IsTrue(left.Shape == right.Shape, "The input tensors must have the same shape");
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x1, Tensor x2)
        {
            var y = x1.Clone();
            CpuBlas.Sum(x2, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x1, Tensor x2, Tensor y, Tensor dy)
        {
            return dy.Clone();
        }
    }
}
