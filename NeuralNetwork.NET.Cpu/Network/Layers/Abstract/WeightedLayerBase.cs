using System;
using System.Security.Cryptography;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Interfaces;
using NeuralNetworkDotNet.APIs.Models;
using NeuralNetworkDotNet.APIs.Structs;
using NeuralNetworkDotNet.Network.Layers.Abstract.Base;

namespace NeuralNetworkDotNet.Network.Layers.Abstract
{
    /// <summary>
    /// The base <see langword="class"/> for all the network layers that have weights and biases as parameters
    /// </summary>
    internal abstract class WeightedLayerBase : LayerBase
    {
        #region Parameters

        /// <summary>
        /// Gets the weights for the current network layer
        /// </summary>
        [NotNull]
        public float[] Weights { get; }

        /// <summary>
        /// Gets the biases for the current network layer
        /// </summary>
        [NotNull]
        public float[] Biases { get; }

        /// <summary>
        /// Gets an SHA256 hash calculated on both the weights and biases of the layer
        /// </summary>
        [NotNull]
        public virtual string Hash => Convert.ToBase64String(Sha256.Hash(Weights, Biases));

        #endregion

        protected WeightedLayerBase(Shape input, Shape output, [NotNull] float[] w, [NotNull] float[] b) : base(input, output)
        {
            Weights = w;
            Biases = b;
        }

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the layer
        /// </summary>
        /// <param name="x">The layer inputs used in the forward pass</param>
        /// <param name="y">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        /// <param name="dx">The resulting backpropagated error</param>
        /// <param name="dJdw">The resulting gradient with respect to the weights</param>
        /// <param name="dJdb">The resulting gradient with respect to the biases</param>
        public abstract void Backpropagate([NotNull] Tensor x, [NotNull] Tensor y, [NotNull] Tensor dy, [NotNull] Tensor dx, out Tensor dJdw, out Tensor dJdb);

        /// <summary>
        /// Checks whether or not all the weights in the current layer are valid and the layer can be safely used
        /// </summary>
        [Pure]
        public virtual bool ValidateWeights()
        {
            if (Weights.AsSpan().HasNaN()) return false;
            if (Biases.AsSpan().HasNaN()) return false;

            return true;
        }

        /// <inheritdoc/>
        public override bool Equals(ILayer other)
        {
            if (!base.Equals(other)) return false;
            return other is WeightedLayerBase layer &&
                   Weights.AsSpan().ContentEquals(layer.Weights) &&
                   Biases.AsSpan().ContentEquals(layer.Biases);
        }
    }
}
