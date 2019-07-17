using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;
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

        private readonly List<Node> _Children = new List<Node>();

        /// <summary>
        /// Gets the list of child nodes for the current instance
        /// </summary>
        [NotNull, ItemNotNull]
        public IReadOnlyList<Node> Children => _Children;

        protected Node(Shape shape) => Shape = shape;

        /// <summary>
        /// Appens the given <see cref="Node"/> as child to the current instance
        /// </summary>
        /// <param name="child">The child <see cref="Node"/> to append</param>
        internal void Append([NotNull] Node child)
        {
            Guard.IsFalse(_Children.Contains(child), nameof(child), "The parent node already contains the given child node");

            _Children.Add(child);
        }

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
