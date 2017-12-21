using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using Newtonsoft.Json;
using System;
using System.IO;
using System.Security.Cryptography;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the network layers that have weights and biases as parameters
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class WeightedLayerBase : NetworkLayerBase
    {
        /// <summary>
        /// Gets an SHA256 hash calculated on both the weights and biases of the layer
        /// </summary>
        [NotNull]
        [JsonProperty(nameof(Hash), Order = 5)]
        public unsafe String Hash
        {
            [Pure]
            get
            {
                fixed (float* pw = Weights, pb = Biases)
                {
                    // Use unmanaged streams to avoid copying the weights and biases
                    int
                        weightsSize = sizeof(float) * Weights.Length,
                        biasesSize = sizeof(float) * Biases.Length;
                    using (UnmanagedMemoryStream
                        weightsStream = new UnmanagedMemoryStream((byte*)pw, weightsSize, weightsSize, FileAccess.Read),
                        biasesStream = new UnmanagedMemoryStream((byte*)pb, biasesSize, biasesSize, FileAccess.Read))
                    using (HashAlgorithm provider = HashAlgorithm.Create(HashAlgorithmName.SHA256.Name))
                    {
                        // Compute the two SHA256 hashes and combine them (there isn't a way to concatenate two streams with the hash class)
                        byte[]
                            weightsHash = provider.ComputeHash(weightsStream),
                            biasesHash = provider.ComputeHash(biasesStream),
                            hash = new byte[32];
                        unchecked
                        {
                            for (int i = 0; i < 32; i++)
                                hash[i] = (byte)((17 * 31 * weightsHash[i] * 31 * biasesHash[i]) % byte.MaxValue); // Trust me
                        }

                        // Convert the final hash to a base64 string
                        return Convert.ToBase64String(hash);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the weights for the current network layer
        /// </summary>
        [NotNull]
        public float[,] Weights { get; }

        /// <summary>
        /// Gets the biases for the current network layer
        /// </summary>
        [NotNull]
        public float[] Biases { get; }

        protected WeightedLayerBase(in TensorInfo input, in TensorInfo output, [NotNull] float[,] w, [NotNull] float[] b, ActivationFunctionType activation) 
            : base(input, output, activation)
        {
            Weights = w;
            Biases = b;
        }

        /// <summary>
        /// Computes the gradient for the weights in the current network layer
        /// </summary>
        /// <param name="a">The input activation</param>
        /// <param name="delta">The output delta</param>
        /// <param name="dJdw">The resulting gradient with respect to the weights</param>
        /// <param name="dJdb">The resulting gradient with respect to the biases</param>
        public abstract void ComputeGradient(in Tensor a, in Tensor delta, out Tensor dJdw, out Tensor dJdb);


        #region Implementation

        /// <summary>
        /// Tweaks the layer weights and biases according to the input gradient and parameters
        /// </summary>
        /// <param name="dJdw">The gradient with respect to the weights</param>
        /// <param name="dJdb">The gradient with respect to the biases</param>
        /// <param name="alpha">The learning rate to use when updating the weights</param>
        /// <param name="l2Factor">The L2 regularization factor to resize the weights</param>
        public unsafe void Minimize(in Tensor dJdw, in Tensor dJdb, float alpha, float l2Factor)
        {
            // Tweak the weights
            fixed (float* pw = Weights)
            {
                float* pdj = dJdw;
                int
                    h = Weights.GetLength(0),
                    w = Weights.GetLength(1);
                for (int x = 0; x < h; x++)
                {
                    int offset = x * w;
                    for (int y = 0; y < w; y++)
                    {
                        int target = offset + y;
                        pw[target] -= l2Factor * pw[target] + alpha * pdj[target];
                    }
                }
            }

            // Tweak the biases of the lth layer
            fixed (float* pb = Biases)
            {
                float* pdj = dJdb;
                int w = Biases.Length;
                for (int x = 0; x < w; x++)
                    pb[x] -= alpha * pdj[x];
            }
        }

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

        #region Equality check

        /// <inheritdoc/>
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            return other is WeightedLayerBase layer &&
                   Weights.ContentEquals(layer.Weights) &&
                   Biases.ContentEquals(layer.Biases);
        }

        #endregion

        /// <inheritdoc/>
        public override void Serialize([NotNull] Stream stream)
        {
            base.Serialize(stream);
            stream.Write(Weights);
            stream.Write(Biases);
        }
    }
}
