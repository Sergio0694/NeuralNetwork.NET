using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;

namespace NeuralNetworkDotNet.Network.Nodes
{
    /// <summary>
    /// A base <see langword="class"/> for all the available nodes
    /// </summary>
    public abstract partial class Node : IEquatable<Node>
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

        /// <summary>
        /// Creates a new <see cref="Node"/> instance with the specified output shape
        /// </summary>
        /// <param name="shape">The output shape for the current <see cref="Node"/></param>
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
    }
}
