using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Misc;

namespace NeuralNetworkNET.Networks.Implementations.Layers.Abstract
{
    /// <summary>
    /// The base class for all the network layers that have weights and biases as parameters
    /// </summary>
    internal abstract class WeightedLayerBase : NetworkLayerBase
    {
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
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public abstract LayerGradient ComputeGradient([NotNull] float[,] a, float[,] delta);

        public unsafe void Minimize(LayerGradient gradient, float alpha, float l2Factor)
        {
            // Tweak the weights
            fixed (float* pw = Weights, pdj = gradient.DJdw)
            {
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
            fixed (float* pb = Biases, pdj = gradient.Djdb)
            {
                int w = Biases.Length;
                for (int x = 0; x < w; x++)
                    pb[x] -= alpha * pdj[x];
            }
        }
    }
}
