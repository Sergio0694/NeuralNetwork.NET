using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Nodes.Interfaces;

namespace NeuralNetworkDotNet.Network.Nodes.Abstract
{
    /// <summary>
    /// A base <see langword="class"/> for all nodes representing binary operations
    /// </summary>
    public abstract class BinaryNode : INode
    {
        /// <summary>
        /// Gets the first parent <see cref="INode"/> instance for the current node
        /// </summary>
        [NotNull]
        public INode LeftParent { get; }

        /// <summary>
        /// Gets the second <see cref="INode"/> instance for the current node
        /// </summary>
        [NotNull]
        public INode RightParent { get; }

        /// <inheritdoc/>
        public Shape Shape { get; }

        /// <summary>
        /// Creates a new <see cref="BinaryNode"/> instance with the specified parameters
        /// </summary>
        /// <param name="left">The left parent <see cref="INode"/></param>
        /// <param name="right">The right parent <see cref="INode"/></param>
        /// <param name="shape">The output <see cref="Shape"/> value for the current node</param>
        protected BinaryNode([NotNull] INode left, [NotNull] INode right, Shape shape)
        {
            Guard.IsTrue(shape.N == -1, nameof(shape), "The output shape can't have a defined N channel");

            LeftParent = left;
            RightParent = right;
            Shape = shape;
        }

        /// <summary>
        /// Forwards the inputs through the layer and returns the resulting <see cref="Tensor"/>
        /// </summary>
        /// <param name="x1">The left input <see cref="Tensor"/> to process</param>
        /// <param name="x2">The right input <see cref="Tensor"/> to process</param>
        [MustUseReturnValue, NotNull]
        public abstract Tensor Forward([NotNull] Tensor x1, [NotNull] Tensor x2);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="x1">The left input <see cref="Tensor"/> used in the forward pass</param>
        /// <param name="x2">The right input <see cref="Tensor"/> used in the forward pass</param>
        /// <param name="y">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        /// <param name="dx1">The backpropagated delta for the left parent node</param>
        /// <param name="dx2">The backpropagated delta for the right parent node</param>
        [MustUseReturnValue, NotNull]
        public abstract Tensor Backward(
            [NotNull] Tensor x1, [NotNull] Tensor x2,
            [NotNull] Tensor y, [NotNull] Tensor dy,
            out Tensor dx1, out Tensor dx2);

        /// <inheritdoc/>
        public bool Equals(INode other)
        {
            if (other == null) return false;

            return GetType() == other.GetType() &&
                   other is BinaryNode binary &&
                   LeftParent.Shape == binary.LeftParent.Shape &&
                   RightParent.Shape == binary.RightParent.Shape &&
                   Shape == binary.Shape;
        }

        public abstract INode Clone();
    }
}
