// ===================================
// Credits for the base of this code
// ===================================
//
// neural-networks-and-deep-learning 
// Michael Nielsen
// https://github.com/mnielsen/neural-networks-and-deep-learning
//
// network.py
// ~~~~~~~~~~
//
// A module to implement the stochastic gradient descent learning
// algorithm for a feedforward neural network.  Gradients are calculated
// using backpropagation.  Note that I have focused on making the code
// simple, easily readable, and easily modifiable.  It is not optimized,
// and omits many desirable features.

using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;

namespace NeuralNetworkNET.Networks.Implementations
{
    public class NumpyNetwork
    {
        public readonly int num_layers;

        public int[] sizes;

        public double[][,] biases;

        public double[][,] weights;

        public NumpyNetwork(params int[] neurons)
        {
            num_layers = neurons.Length;
            sizes = neurons;
            var r = new Random();
            biases = neurons.Skip(1).Select(n => r.NextGaussianMatrix(n, 1)).ToArray();
            weights = neurons.Take(neurons.Length - 1).Select((n, i) => r.NextGaussianMatrix(n, neurons[i + 1])).ToArray();
        }

        public double[,] feedforward(double[,] a)
        {
            foreach ((var b, var w) in biases.Zip(weights, (b, w) => (b, w)))
            {
                var mul = w.Multiply(a);
                var sum = mul.Sum(b);
                a = sum.Activation(ActivationFunctions.Sigmoid);
            }
            return a;
        }

        public void SGD(IReadOnlyList<(double[,], double[,])> training_data, int epochs, int mini_batch_size, double eta, IReadOnlyList<(double[,], double)> test_data)
        {
            var n_test = test_data.Count;
            var n = training_data.Count;
            foreach (var j in Enumerable.Range(0, epochs))
            {
                var random = new Random();
                training_data = training_data.OrderBy(_ => random.Next()).ToArray();
                var mini_batches = Enumerable.Range(0, n / mini_batch_size).Select(i => training_data.Skip(i * mini_batch_size).Take(mini_batch_size).ToArray()).ToArray();
                foreach (var mini_batch in mini_batches)
                    update_mini_batch(mini_batch, eta);
                Console.WriteLine($"Epoch {j}: {evaluate(test_data)} / {n_test}");
            }
        }

        private void update_mini_batch(IReadOnlyList<(double[,], double[,])> mini_batch, double eta)
        {
            var nabla_b = biases.Select(b => new double[b.Length, 1]).ToArray();
            var nabla_w = weights.Select(w => new double[w.GetLength(0), w.GetLength(1)]).ToArray();
            foreach ((var x, var y) in mini_batch)
            {
                (var delta_nabla_b, var delta_nabla_w) = backprop(x, y);
                nabla_b = nabla_b.Zip(delta_nabla_b, (nb, dnb) => nb.Sum(dnb)).ToArray();
                nabla_w = nabla_w.Zip(delta_nabla_w, (nw, dnw) => nw.Sum(dnw)).ToArray();
            }

            weights = weights.Zip(nabla_w, (w, nw) =>
            {
                nw.Tweak(d => -(eta / mini_batch.Count * d));
                return w.Sum(nw);
            }).ToArray();
            biases = biases.Zip(nabla_b, (b, nb) =>
            {
                nb.Tweak(d => -(eta / mini_batch.Count * d));
                return b.Sum(nb);
            }).ToArray();
        }

        private (double[][,], double[][,]) backprop(double[,] x, double[,] y)
        {
            var nabla_b = biases.Select(b => new double[b.Length,1]).ToArray();
            var nabla_w = weights.Select(w => new double[w.GetLength(0), w.GetLength(1)]).ToArray();
            var activation = x;
            var activations = new List<double[,]> { x };
            var zs = new List<double[,]>();

            foreach ((var b, var w) in biases.Zip(weights, (b, w) => (b, w)))
            {
                var z0 = w.Multiply(activation);
                var z = z0.Sum(b);
                zs.Add(z);
                activation = z.Activation(ActivationFunctions.Sigmoid);
                activations.Add(activation);
            }

            var cost = cost_derivative(activations.Last(), y);
            var delta = cost.HadamardProduct(zs.Last().Activation(ActivationFunctions.SigmoidPrime));
            nabla_b[nabla_b.Length - 1] = delta;
            nabla_w[nabla_w.Length - 1] = delta.Multiply(activations[activations.Count - 2].Transpose());

            foreach (var l in Enumerable.Range(2, num_layers))
            {
                var z = zs[zs.Count - l];
                var sp = z.Activation(ActivationFunctions.SigmoidPrime);
                delta = weights[zs.Count - l + 1].Transpose().Multiply(delta).HadamardProduct(sp);
                nabla_b[nabla_b.Length - l] = delta;
                nabla_w[nabla_w.Length - l] = delta.Multiply(activations[activations.Count - l - 1].Transpose());
            }

            return (nabla_b, nabla_w);
        }

        private object evaluate(IReadOnlyList<(double[,], double)> test_data)
        {
            var test_results = test_data.Select(tuple => (feedforward(tuple.Item1).Argmax(), tuple.Item2));
            return test_results.Count(tuple => ((double)tuple.Item1).EqualsWithDelta(tuple.Item2));
        }

        private double[,] cost_derivative(double[,] output_activations, double[,] y)
        {
            return output_activations.Subtract(y);
        }
    }
}
