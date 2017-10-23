using System;
using System.ComponentModel;
using System.Threading;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.SupervisedLearning
{
    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class GradientDescentNetworkTrainer
    {
        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="size">The number of nodes in the hidden layer of the network (it will be decided automatically if null)</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="solution">An optional starting solution to resume a previous training session</param>
        /// <param name="progress">An optional progress callback</param>
        [PublicAPI]
        [Pure]
        [ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static async Task<SingleLayerPerceptron> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys, [CanBeNull] int? size,
            CancellationToken token,
            [CanBeNull] double[] solution = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress = null)
        {
            // Preliminary checks
            if (x.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (x.GetLength(0) != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (size <= 0) throw new ArgumentOutOfRangeException("The hidden layer must have a positive number of nodes");

            // Calculate the target network size
            int
                inputs = x.GetLength(1),
                outputs = ys.GetLength(1);
            int iSize = size ?? (inputs + outputs) / 2;

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] w1w2)
            {
                SingleLayerPerceptron network = SingleLayerPerceptron.Deserialize(inputs, iSize, outputs, w1w2);
                return network.CalculateCost(x, ys);
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] w1w2)
            {
                SingleLayerPerceptron network = SingleLayerPerceptron.Deserialize(inputs, iSize, outputs, w1w2);
                return network.CostFunctionPrime(x, ys);
            }

            // Initialize the optimization function
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(
                inputs * iSize + iSize * outputs, // Number of free variables in the function to optimize
                CostFunction, GradientFunction)
            {
                Token = token
            };

            // Handle the progress if necessary
            if (progress != null) bfgs.Progress += (s, e) =>
            {
                if (double.IsNaN(e.Value)) return;
                progress.Report(new BackpropagationProgressEventArgs(
                    () => SingleLayerPerceptron.Deserialize(inputs, iSize, outputs, e.Solution), e.Iteration, e.Value));
            };

            // Minimize the cost function
            await Task.Run(() =>
            {
                if (solution != null) bfgs.Minimize(solution);
                else bfgs.Minimize();
            }, token);

            // Return the result network
            return SingleLayerPerceptron.Deserialize(inputs, iSize, outputs, bfgs.Solution);
        }

        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="solution">An optional starting solution to resume a previous training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <param name="neurons">The number of neurons in each network layer</param>
        [PublicAPI]
        [Pure]
        [ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static async Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            CancellationToken token,
            [CanBeNull] double[] solution,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress,
            [NotNull] params int[] neurons)
        {
            // Preliminary checks
            if (x.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (x.GetLength(0) != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (neurons.Length < 2) throw new ArgumentOutOfRangeException("The network must have at least two layers");

            // Calculate the target network size
            int size = 0;
            for (int i = 0; i < neurons.Length - 1; i++)
                size += neurons[i] * neurons[i + 1];

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                return network.CalculateCost(x, ys);
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                return network.ComputeGradient(x, ys);
            }

            // Initialize the optimization function
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(
                size, // Number of free variables in the function to optimize
                CostFunction, GradientFunction)
            {
                Token = token
            };

            // Handle the progress if necessary
            if (progress != null) bfgs.Progress += (s, e) =>
            {
                if (double.IsNaN(e.Value)) return;
                progress.Report(new BackpropagationProgressEventArgs(
                    () => NeuralNetwork.Deserialize(e.Solution, neurons), e.Iteration, e.Value));
            };

            // Minimize the cost function
            await Task.Run(() =>
            {
                if (solution != null) bfgs.Minimize(solution);
                else bfgs.Minimize();
            }, token);

            // Return the result network
            return NeuralNetwork.Deserialize(bfgs.Solution, neurons);
        }

        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="solution">An optional starting solution to resume a previous training session</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="progress">An optional progress callback</param>
        /// <param name="neurons">The number of neurons in each network layer</param>
        [PublicAPI]
        [Pure]
        [ItemNotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static async Task<INeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys,
            [CanBeNull] double[] solution,
            CancellationToken token,
            [CanBeNull] IProgress<ProgressChangedEventArgs> progress,
            [NotNull] params int[] neurons)
        {
            // Preliminary checks
            if (x.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (x.GetLength(0) != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (neurons.Length < 2) throw new ArgumentOutOfRangeException("The network must have at least two layers");

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                return network.CalculateCost(x, ys);
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] weights)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(weights, neurons);
                return network.ComputeGradient(x, ys);
            }

            // Initialize the optimization function
            GradientDescent gd = new GradientDescent
            {
                Function = CostFunction,
                Gradient = GradientFunction,
                Token = token
            };

            // Handle the progress if necessary
            if (progress != null) gd.ProgressChanged += (s, e) => progress.Report(e);

            // Minimize the cost function
            await Task.Run(() =>
            {
                if (solution != null) gd.Minimize(solution);
                else
                {
                    NeuralNetwork network = NeuralNetwork.NewRandom(neurons);
                    double[] start = network.Serialize();
                    gd.Minimize(start);
                }
            }, token);

            // Return the result network
            return NeuralNetwork.Deserialize(gd.Solution, neurons);
        }
    }
}
