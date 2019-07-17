using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.cpuDNN;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Nodes.Abstract;
using NeuralNetworkDotNet.Network.Nodes.Enums;

namespace NeuralNetworkDotNet.Network.Nodes.Unary
{
    /// <summary>
    /// A dropout node, used to improve the training in deep networks
    /// </summary>
    internal sealed class DropoutNode : UnaryNodeBase
    {
        /// <inheritdoc/>
        public override NodeType Type => NodeType.Dropout;

        public DropoutNode([NotNull] Node input) : base(input, input.Shape) { }

        [CanBeNull]
        private Tensor _Mask;

        /// <inheritdoc/>
        public override Tensor Forward(Tensor x)
        {
            // Create the dropout mask, if needed
            if (_Mask?.Shape != x.Shape)
            {
                _Mask?.Dispose();
                _Mask = Tensor.Like(x);
            }

            // TODO: handle inference mode and variable factor
            var y = Tensor.Like(x);
            CpuDnn.DropoutForward(0.6f, x, y, _Mask);

            return y;
        }

        /// <inheritdoc/>
        public override Tensor Backward(Tensor x, Tensor y, Tensor dy)
        {
            Guard.IsFalse(_Mask == null, "There isn't a valid dropout mask to use");
            Guard.IsTrue(_Mask.Shape == x.Shape, "The stored dropout mask doesn't have a valid shape");

            var dx = Tensor.Like(x);
            CpuBlas.MultiplyElementwise(dy, _Mask, dx);

            return dx;
        }
    }
}
