using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs.Info;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Network.Nodes.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// A pooling node, with a 2x2 window and a stride of 2
    /// </summary>
    internal sealed class PoolingNode : UnaryNodeBase
    {
        private readonly PoolingInfo _OperationInfo;

        /// <summary>
        /// Gets the info on the pooling operation performed by the layer
        /// </summary>
        public ref readonly PoolingInfo OperationInfo
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _OperationInfo;
        }

        public PoolingNode([NotNull] Node input, PoolingInfo operation) : base(input, operation.GetOutputShape(input.Shape))
        {
            _OperationInfo = operation;
        }

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            var y = Tensor.New(x.Shape.N, Shape.C, Shape.H, Shape.W);
            CpuDnn.PoolingForward(x, y);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            var dx = Tensor.Like(x);
            CpuDnn.PoolingBackward(x, dy, dx);

            return dx;
        }

        /// <inheritdoc/>
        public override bool Equals(Node other)
        {
            if (!base.Equals(other)) return false;

            return other is PoolingNode node &&
                   OperationInfo.Equals(node.OperationInfo);
        }
    }
}