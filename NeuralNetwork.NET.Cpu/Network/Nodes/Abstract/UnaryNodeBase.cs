using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.Network.Nodes.Abstract
{
    /// <summary>
    /// A base <see langword="class"/> for all nodes representing unary operations
    /// </summary>
    public abstract class UnaryNodeBase : INode
    {
        /// <summary>
        /// Gets the parent <see cref="INode"/> instance for the current node
        /// </summary>
        [NotNull]
        public INode Parent { get; }

        /// <inheritdoc/>
        public Shape Shape { get; }

        /// <summary>
        /// Creates a new <see cref="UnaryNodeBase"/> instance with the specified parameters
        /// </summary>
        /// <param name="input">The input <see cref="INode"/> instance</param>
        /// <param name="shape">The output <see cref="Shape"/> value for the current node</param>
        protected UnaryNodeBase([NotNull] INode input, Shape shape)
        {
            Guard.IsTrue(shape.N == -1, nameof(shape), "The output shape can't have a defined N channel");

            Parent = input;
            Shape = shape;
        }

        /// <summary>
        /// Forwards the inputs through the layer and returns the resulting <see cref="Tensor"/>
        /// </summary>
        /// <param name="x">The input <see cref="Tensor"/> to process</param>
        [MustUseReturnValue, NotNull]
        public abstract Tensor Forward([NotNull] Tensor x);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="x">The layer inputs used in the forward pass</param>
        /// <param name="y">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        [MustUseReturnValue, NotNull]
        public abstract Tensor Backward([NotNull] Tensor x, [NotNull] Tensor y, [NotNull] Tensor dy);

        /// <inheritdoc/>
        public virtual bool Equals(INode other)
        {
            if (other == null) return false;

            return GetType() == other.GetType() &&
                   other is UnaryNodeBase unary &&
                   Parent.Shape == unary.Parent.Shape &&
                   Shape == unary.Shape;
        }
    }
}
