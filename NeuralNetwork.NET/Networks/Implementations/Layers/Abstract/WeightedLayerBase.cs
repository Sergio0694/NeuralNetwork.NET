using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Structs;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the network layers that have weights and biases as parameters
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class WeightedLayerBase : NetworkLayerBase
    {
        /// <summary>
        /// Gets the weights for the current network layer
        /// </summary>
        [NotNull]
        [JsonProperty(nameof(Weights), Required = Required.Always, Order = 10)]
        public float[,] Weights { get; }

        /// <summary>
        /// Gets the biases for the current network layer
        /// </summary>
        [NotNull]
        [JsonProperty(nameof(Biases), Required = Required.Always, Order = 11)]
        public float[] Biases { get; }

        protected WeightedLayerBase([NotNull] float[,] w, [NotNull] float[] b, ActivationFunctionType activation) : base(activation)
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
        public abstract void ComputeGradient(in FloatSpan2D a, in FloatSpan2D delta, out FloatSpan2D dJdw, out FloatSpan dJdb);


        #region Implementation

        /// <summary>
        /// Tweaks the layer weights and biases according to the input gradient and parameters
        /// </summary>
        /// <param name="dJdw">The gradient with respect to the weights</param>
        /// <param name="dJdb">The gradient with respect to the biases</param>
        /// <param name="alpha">The learning rate to use when updating the weights</param>
        /// <param name="l2Factor">The L2 regularization factor to resize the weights</param>
        public unsafe void Minimize(in FloatSpan2D dJdw, in FloatSpan dJdb, float alpha, float l2Factor)
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
    }
}
