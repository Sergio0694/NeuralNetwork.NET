using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Networks.Implementations
{
    internal sealed class NeuralNetwork
    {
        #region Fields and parameters

        private readonly IReadOnlyList<double[,]> Weights;

        private readonly IReadOnlyList<double[]> Biases;

        private readonly IReadOnlyList<double[,]> TransposedWeights;

        #endregion

        /// <summary>
        /// Initializes a new instance with the given parameters
        /// </summary>
        /// <param name="weights">The weights in all the network layers</param>
        /// <param name="biases">The bias vectors to use in the network</param>
        internal NeuralNetwork([NotNull] IReadOnlyList<double[,]> weights, [NotNull] IReadOnlyList<double[]> biases)
        {
            // Input check
            if (weights.Count == 0) throw new ArgumentOutOfRangeException(nameof(weights), "The weights must have a length at least equal to 1");
            if (biases.Count != weights.Count) throw new ArgumentException(nameof(biases), "The bias vector has an invalid size");
            for (int i = 0; i < weights.Count; i++)
            {
                if (i > 0 && weights[i - 1].GetLength(1) != weights[i].GetLength(0))
                    throw new ArgumentOutOfRangeException(nameof(weights), "Some weight matrix doesn't have the right size");
                if (weights[i].GetLength(1) != biases[i].Length)
                    throw new ArgumentException(nameof(biases), $"The bias vector #{i} doesn't have the right size");
            }

            // Parameters setup
            Weights = weights;
            TransposedWeights = Weights.Select(m => m.Transpose()).ToArray();
            Biases = biases;
        }

        /// <summary>
        /// Creates a new random instance with the given number of neurons in each layer
        /// </summary>
        /// <param name="neurons">The number of neurons from the input to the output layer</param>
        [NotNull]
        internal static NeuralNetwork NewRandom([NotNull] params int[] neurons)
        {
            if (neurons.Length < 2) throw new ArgumentOutOfRangeException(nameof(neurons), "The network must have at least two layers");
            Random random = new Random();
            double[][,] weights = new double[neurons.Length - 1][,];
            double[][] biases = new double[neurons.Length - 1][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextMatrix(neurons[i], neurons[i + 1]);
                double[] bias = new double[neurons[i]];
                for (int j = 0; j < neurons[i]; j++)
                    bias[j] = random.NextGaussian();
                biases[i] = bias;
            }
            return new NeuralNetwork(weights, biases);
        }

        #region Single processing

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[] Forward([NotNull] double[] x)
        {
            if (x.Length == 0) throw new ArgumentException(nameof(x), "The input array can't be empty");
            double[,] temp = new double[1, x.Length];
            Buffer.BlockCopy(x, 0, temp, 0, sizeof(double) * x.Length);
            double[,] yHat = Forward(temp);
            double[] output = new double[yHat.GetLength(1)];
            Buffer.BlockCopy(yHat, 0, output, 0, sizeof(double) * output.Length);
            return output;
        }

        [PublicAPI]
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        internal double[] CostFunctionPrime(double[] input, double[] y)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region Batch processing

        [PublicAPI]
        [MustUseReturnValue]
        [CollectionAccess(CollectionAccessType.Read)]
        public double[,] Forward([NotNull] double[,] x)
        {
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                double[,] zi = a0.Multiply(Weights[i]); // W(l) * A(l - 1)
                zi.SumSE(Biases[i]);                    // Z(l) =  W(l) * A(l - 1) + B(l)
                zi.SigmoidSE();                         // A(l) = sigm(Z(l))
                a0 = zi;
            }
            return a0; // At least one weight matrix, so a0 != x
        }

        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal double[] CostFunctionPrime(double[,] x, double[,] y)
        {
            // Feedforward
            int steps = Weights.Count - 1;  // Number of forward hops through the network
            double[][,] 
                zList = new double[steps][,],
                aList = new double[steps][,];
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                double[,] zi = a0.Multiply(Weights[i]);
                zi.SumSE(Biases[i]);
                zList[i] = zi;
                aList[i] = a0 = zi.Sigmoid();
            }

            // Output error d(L)
            double[,]
                zLPrime = zList[zList.Length - 1].SigmoidPrime(),   // Sigmoid prime of zL
                gA = aList[aList.Length - 1].Subtract(y),           // Gradient of C with respect to a, so (yHat - y)
                dL = gA.HadamardProduct(zLPrime);                   // dL, Hadamard product of the gradient and the sigmoid prime for L

            // Backpropagation
            double[][,] deltas = new double[steps][,];
            for (int l = Weights.Count - 2; l >= 1; l--)    // Loop for l = L - 1, L - 2, ..., 2
            {
                double[,]
                    dleft = TransposedWeights[l].Multiply(l == Weights.Count - 2 ? dL : deltas[l - 1]),
                    dPrime = zList[l].SigmoidPrime(),
                    dl = dleft.HadamardProduct(dPrime);
                deltas[l] = dl;
            }

            // Compute the gradient
            int dLength = Weights.Count + deltas.Sum(d => d.Length) + 1;
            double[] gradient = new double[dLength];
            int position = 0;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Compute dJdw(l)
                double[,] dJdw;
                if (i == 0) dJdw = x.Multiply(deltas[i]);
                else if (i == Weights.Count - 1) dJdw = aList[i - 1].Multiply(dL);
                else dJdw = aList[i - 1].Multiply(deltas[i]);

                // Populate the gradient vector
                int bytes = sizeof(double) * dJdw.Length;
                Buffer.BlockCopy(dJdw, 0, gradient, position, bytes);
                position += bytes;
                double[,] di = i == Weights.Count - 1 ? dL : deltas[i];
                bytes = sizeof(double) * di.Length;
                Buffer.BlockCopy(di, 0, gradient, position, bytes);
                position += bytes;
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
        internal static NeuralNetwork Deserialize([NotNull] double[] data, [NotNull] params int[] neurons)
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

            // Create the new network to use
            return new NeuralNetwork(weights, biases);
        }

        [PublicAPI]
        [Pure]
        internal double[] SerializeWeights()
        {
            int length = Weights.Sum(layer => layer.Length) + Biases.Sum(bias => bias.Length);
            double[] weights = new double[length];
            int position = 0;
            for (int i = 0; i < Weights.Count; i++)
            {
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
        internal NeuralNetwork Crossover(NeuralNetwork other, Random random)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
