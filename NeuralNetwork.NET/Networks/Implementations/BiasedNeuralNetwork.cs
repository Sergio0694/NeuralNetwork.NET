using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.PublicAPIs;
using Newtonsoft.Json;
#pragma warning disable 162 // TODO: remove after implementing the gradient function

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A complete and fully connected neural network with an arbitrary number of hidden layers, with additional bias vectors
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal sealed class BiasedNeuralNetwork : NeuralNetwork
    {
        #region Local fields

        /// <summary>
        /// The list of bias vectors for the network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Biases), Required = Required.Always)]
        private readonly IReadOnlyList<double[]> Biases;

        #endregion

        #region Initialization

        /// <summary>
        /// Initializes a new instance with the given parameters
        /// </summary>
        /// <param name="weights">The weights in all the network layers</param>
        /// <param name="biases">The bias vectors to use in the network</param>
        internal BiasedNeuralNetwork([NotNull] IReadOnlyList<double[,]> weights, [NotNull] IReadOnlyList<double[]> biases) : base(weights)
        {
            // Input check
            if (biases.Count != weights.Count) throw new ArgumentException(nameof(biases), "The bias vector has an invalid size");
            for (int i = 0; i < weights.Count; i++)
            {
                if (weights[i].GetLength(1) != biases[i].Length)
                    throw new ArgumentException(nameof(biases), $"The bias vector #{i} doesn't have the right size");
            }

            // Parameters setup
            Biases = biases;
        }

        /// <summary>
        /// Creates a new random instance with the given number of neurons in each layer
        /// </summary>
        /// <param name="neurons">The number of neurons from the input to the output layer</param>
        [NotNull]
        internal new static BiasedNeuralNetwork NewRandom([NotNull] params int[] neurons)
        {
            if (neurons.Length < 2) throw new ArgumentOutOfRangeException(nameof(neurons), "The network must have at least two layers");
            Random random = new Random();
            double[][,] weights = new double[neurons.Length - 1][,];
            double[][] biases = new double[neurons.Length - 1][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextGaussianMatrix(neurons[i], neurons[i + 1]);
                int next = neurons[i + 1];
                double[] bias = new double[next];
                for (int j = 0; j < next; j++)
                    bias[j] = random.NextGaussian();
                biases[i] = bias;
            }
            return new BiasedNeuralNetwork(weights, biases);
        }

        #endregion

        #region Batch processing

        /// <inheritdoc/>
        public override double[,] Forward(double[,] x)
        {
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // A(l) = sigm(W(l) * A(l - 1) + b(l))
                a0 = a0.MultiplyWithSumAndSigmoid(Weights[i], Biases[i]);
            }
            return a0; // At least one weight matrix, so a0 != x
        }

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="y">The expected results</param>
        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal override double[] ComputeGradient(double[,] x, double[,] y)
        {
            // Feedforward
            int steps = Weights.Count;  // Number of forward hops through the network
            double[][,]
                zList = new double[steps][,],
                aList = new double[steps][,];
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Save the intermediate steps to be able to reuse them later
                double[,] zi = a0.MultiplyWithSum(Weights[i], Biases[i]);
                zList[i] = zi;
                aList[i] = a0 = zi.Sigmoid();
            }

            /* ============================
             * Calculate delta(L) in place
             * ============================
             * Perform the sigmoid prime of zL, the activity on the last layer
             * Calculate the gradient of C with respect to a, so (yHat - y)
             * Compute d(L), the Hadamard product of the gradient and the sigmoid prime for L */
            double[,] dL = aList[aList.Length - 1];
            dL.InPlaceSubtractAndHadamardProductWithSigmoidPrime(y, zList[zList.Length - 1]);

            // Backpropagation
            double[][,] deltas = new double[steps][,];      // One additional delta for each hop, delta(L) has already been calculated
            deltas[steps - 1] = dL;                         // Store the delta(L) in the last position
            for (int l = Weights.Count - 2; l >= 0; l--)    // Loop for l = L - 1, L - 2, ..., 2
            {
                // Precompute  W(l + 1) * delta(l + 1)
                double[,]
                    dleft = deltas[l + 1].Multiply(TransposedWeights[l + 1]),
                    dl = zList[l]; // Local reference on the delta to calculate in place

                /* ============================
                 * Calculate delta(l) in place
                 * ============================
                 * Perform the sigmoid prime of z(l), the activity on the previous layer
                 * Compute d(l), the Hadamard product of z'(l) and W(l + 1) * delta(l + 1) */
                dl.InPlaceSigmoidPrimeAndHadamardProduct(dleft);
                deltas[l] = dl;
            }

            // Compute the gradient
            int dLength = Weights.Sum(w => w.Length) + deltas.Sum(d => d.Length);
            double[] gradient = new double[dLength];
            int position = 0;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Store the target delta
                double[,] di = deltas[i];

                // Compute dJdw(l)
                double[,] dJdw = i == 0 
                    ? x.Transpose().Multiply(di)                // dJdW1, transposed input * first delta
                    : aList[i - 1].Transpose().Multiply(di);    // dJdWi, previous activation transposed * current delta

                // Populate the gradient vector
                int bytes = sizeof(double) * dJdw.Length;
                Buffer.BlockCopy(dJdw, 0, gradient, position, bytes);
                position += bytes;
                
                // Handle the gradient with respect to the current bias vector
                throw new NotImplementedException();
            }
            return gradient;
        }

        #endregion

        #region Tools

        /// <summary>
        /// Deserializes a neural network from the input weights and parameters
        /// </summary>
        /// <param name="data">The data representing the weights and the biases of the network</param>
        /// <param name="neurons">The number of nodes in each network layer</param>
        [PublicAPI]
        [Pure, NotNull]
        internal new static BiasedNeuralNetwork Deserialize([NotNull] double[] data, [NotNull] params int[] neurons)
        {
            // Checks
            if (neurons.Length < 2) throw new ArgumentException("The network must have at least 2 layers");

            // Parse the input data
            int depth = neurons.Length - 1;
            double[][,] weights = new double[depth][,];
            double[][] biases = new double[depth][];
            int position = 0;
            for (int i = 0; i < depth; i++)
            {
                // Unpack the current weights
                double[,] wi = new double[neurons[i], neurons[i + 1]];
                int bytes = sizeof(double) * wi.Length;
                Buffer.BlockCopy(data, position, wi, 0, bytes);
                position += bytes;
                weights[i] = wi;

                // Unpack the current bias vector
                double[] bias = new double[neurons[i + 1]];
                bytes = sizeof(double) * bias.Length;
                Buffer.BlockCopy(data, position, bias, 0, bytes);
                position += bytes;
                biases[i] = bias;
            }
            if (position / sizeof(double) != data.Length) throw new InvalidOperationException("Invalid network requested size");

            // Create the new network to use
            return new BiasedNeuralNetwork(weights, biases);
        }

        /// <summary>
        /// Serializes the current network into a binary representation
        /// </summary>
        /// <returns>A <see cref="double"/> array containing all the weights and biases of the network</returns>
        [PublicAPI]
        [Pure]
        internal override double[] Serialize()
        {
            // Allocate the output array
            int length = Weights.Sum(layer => layer.Length) + Biases.Sum(bias => bias.Length);
            double[] weights = new double[length];
            int position = 0;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Populate the return array with the weights and biases for each layer
                int bytes = sizeof(double) * Weights[i].Length;
                Buffer.BlockCopy(Weights[i], 0, weights, position, bytes);
                position += bytes;
                bytes = sizeof(double) * Biases[i].Length;
                Buffer.BlockCopy(Biases[i], 0, weights, position, bytes);
                position += bytes;
            }
            return weights;
        }

        // Creates a new instance from another network with the same structure
        [Pure]
        internal override NeuralNetwork Crossover(NeuralNetwork other, Random random)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public override bool Equals(INeuralNetwork other)
        {
            // Compare general features
            if (other is BiasedNeuralNetwork network &&
                other.InputLayerSize == InputLayerSize &&
                other.OutputLayerSize == OutputLayerSize &&
                other.HiddenLayers.SequenceEqual(HiddenLayers))
            {
                // Compare each weight and bias value
                for (int i = 0; i < Weights.Count; i++)
                    if (!(network.Weights[i].ContentEquals(Weights[i]) &&
                          network.Biases[i].ContentEquals(Biases[i]))) return false;
                return true;
            }
            return false;
        }

        #endregion
    }
}
