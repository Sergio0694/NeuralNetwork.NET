using System;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Nodes.Binary;

namespace NeuralNetworkDotNet.Network.Nodes
{
    /// <summary>
    /// A base <see langword="class"/> for all the available nodes
    /// </summary>
    public abstract class Node : IEquatable<Node>
    {
        /// <summary>
        /// Gets the shape of the node outputs
        /// </summary>
        public Shape Shape { get; }

        protected Node(Shape shape) => Shape = shape;

        /// <inheritdoc/>
        public virtual bool Equals(Node other)
        {
            if (other == null) return false;
            if (ReferenceEquals(other, this)) return true;

            return GetType() == other.GetType() &&
                   other is Node node &&
                   Shape == node.Shape;
        }

        /// <summary>
        /// Sums the two input nodes and returns a new <see cref="Node"/> instance
        /// </summary>
        /// <param name="a">The first <see cref="Node"/> to sum</param>
        /// <param name="b">The second <see cref="Node"/> to sum</param>
        /// <returns>A new <see cref="Node"/> that performs the sum of the two input nodes</returns>
        [Pure, NotNull]
        public static Node operator +([NotNull] Node a, [NotNull] Node b) => new SumNode(a, b);

        /// <summary>
        /// Stacks (depth concatenation) the two input nodes and returns a new <see cref="Node"/> instance
        /// </summary>
        /// <param name="a">The first <see cref="Node"/> to stack</param>
        /// <param name="b">The second <see cref="Node"/> to stack</param>
        /// <returns>A new <see cref="Node"/> that performs the stack of the two input nodes</returns>
        [Pure, NotNull]
        public static Node operator |([NotNull] Node a, [NotNull] Node b) => new DepthConcatenationNode(a, b);
    }
}
