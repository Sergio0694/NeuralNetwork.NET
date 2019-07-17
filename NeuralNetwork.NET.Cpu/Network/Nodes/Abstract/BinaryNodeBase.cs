using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.Network.Nodes.Abstract
{
    /// <summary>
    /// A base <see langword="class"/> for all nodes representing binary operations
    /// </summary>
    public abstract class BinaryNodeBase : Node
    {
        /// <summary>
        /// Gets the first parent <see cref="Node"/> instance for the current node
        /// </summary>
        [NotNull]
        public Node LeftParent { get; }

        /// <summary>
        /// Gets the second <see cref="Node"/> instance for the current node
        /// </summary>
        [NotNull]
        public Node RightParent { get; }

        /// <summary>
        /// Creates a new <see cref="BinaryNodeBase"/> instance with the specified parameters
        /// </summary>
        /// <param name="left">The left parent <see cref="Node"/></param>
        /// <param name="right">The right parent <see cref="Node"/></param>
        /// <param name="shape">The output <see cref="Shape"/> value for the current node</param>
        protected BinaryNodeBase([NotNull] Node left, [NotNull] Node right, Shape shape) : base(shape)
        {
            Guard.IsTrue(shape.N == -1, nameof(shape), "The output shape can't have a defined N channel");

            LeftParent = left;
            RightParent = right;

            left.Append(this);
            right.Append(this);
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
        [MustUseReturnValue, NotNull]
        public abstract Tensor Backward(
            [NotNull] Tensor x1, [NotNull] Tensor x2,
            [NotNull] Tensor y, [NotNull] Tensor dy);
    }
}
