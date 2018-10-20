using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Abstract
{
    /// <summary>
    /// The base class for all the network layers that have weights and biases as parameters
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class WeightedLayerBase : NetworkLayerBase
    {
        #region Parameters

        /// <summary>
        /// Gets an SHA256 hash calculated on both the weights and biases of the layer
        /// </summary>
        [NotNull]
        [JsonProperty(nameof(Hash), Order = 5)]
        public virtual string Hash => Convert.ToBase64String(Sha256.Hash(Weights, Biases));

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

        #endregion

        protected WeightedLayerBase(in TensorInfo input, in TensorInfo output, [NotNull] float[] w, [NotNull] float[] b, ActivationType activation) 
            : base(input, output, activation)
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
        public abstract void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb);

        #region Implementation

        /// <summary>
        /// Checks whether or not all the weights in the current layer are valid and the layer can be safely used
        /// </summary>
        [Pure]
        [AssertionMethod]
        public unsafe bool ValidateWeights()
        {
            int l = Weights.Length;
            fixed (float* pw = Weights)
                for (int w = 0; w < l; w++)
                    if (float.IsNaN(pw[w]))
                        return false;
            l = Biases.Length;
            fixed (float* pb = Biases)
                for (int w = 0; w < l; w++)
                    if (float.IsNaN(pb[w]))
                        return false;
            return true;
        }

        #endregion

        /// <inheritdoc/>
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            return other is WeightedLayerBase layer &&
                   Weights.ContentEquals(layer.Weights) &&
                   Biases.ContentEquals(layer.Biases);
        }

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(Weights.Length);
            stream.WriteShuffled(Weights);
            stream.Write(Biases.Length);
            stream.WriteShuffled(Biases);
        }
    }
}
