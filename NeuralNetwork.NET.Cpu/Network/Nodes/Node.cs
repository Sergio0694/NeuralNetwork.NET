using System;
using NeuralNetworkDotNet.APIs.Structs;

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
    }
}
