using System;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Helpers;
using NeuralNetworkDotNet.Network.Nodes.Abstract;

namespace NeuralNetworkDotNet.Network.Nodes.Unary.Abstract
{
    /// <summary>
    /// The base <see langword="class"/> for all the network layers that have weights and biases as parameters
    /// </summary>
    internal abstract class WeightedUnaryNodeBase : UnaryNodeBase
    {
        /// <summary>
        /// Gets the weights for the current network layer
        /// </summary>
        [NotNull]
        public Tensor Weights { get; }

        /// <summary>
        /// Gets the biases for the current network layer
        /// </summary>
        [NotNull]
        public Tensor Biases { get; }

        /// <summary>
        /// Gets an SHA256 hash calculated on both the weights and biases of the layer
        /// </summary>
        [NotNull]
        public virtual string Hash => Sha256.Hash(Weights.Span).And(Biases.Span).ToString();

        protected WeightedUnaryNodeBase([NotNull] Node input, Shape shape, [NotNull] Tensor w, [NotNull] Tensor b) : base(input, shape)
        {
            Weights = w;
            Biases = b;
        }

        /// <summary>
        /// Calculates the gradient for the weights in the current layer
        /// </summary>
        /// <param name="x">The layer inputs used in the forward pass</param>
        /// <param name="dy">The output error delta for the current layer</param>
        /// <param name="dJdw">The resulting gradient with respect to the weights</param>
        /// <param name="dJdb">The resulting gradient with respect to the biases</param>
        public abstract void Gradient([NotNull] Tensor x, [NotNull] Tensor dy, out Tensor dJdw, out Tensor dJdb);

        /// <summary>
        /// Checks whether or not all the weights in the current layer are valid and the layer can be safely used
        /// </summary>
        [Pure]
        public virtual bool ValidateWeights() => !(Weights.Span.HasNaN() || Biases.Span.HasNaN());

        /// <inheritdoc/>
        public override bool Equals(Node other)
        {
            if (!base.Equals(other)) return false;

            return other is WeightedUnaryNodeBase layer &&
                   Weights.Equals(layer.Weights) &&
                   Biases.Equals(layer.Biases);
        }
    }
}
