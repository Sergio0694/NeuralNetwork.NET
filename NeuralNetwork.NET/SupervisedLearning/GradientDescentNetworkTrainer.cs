using System;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Implementations;

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
        public static async Task<NeuralNetwork> ComputeTrainedNetworkAsync(
            [NotNull] double[,] x,
            [NotNull] double[,] ys, [CanBeNull] int? size,
            CancellationToken token,
            [CanBeNull] double[] solution = null,
            [CanBeNull] IProgress<BackpropagationProgressEventArgs> progress = null)
        {
            // Preliminary checks
            if (x.Length == 0) throw new ArgumentOutOfRangeException("The input matrix is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (x.Length != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (size <= 0) throw new ArgumentOutOfRangeException("The hidden layer must have a positive number of nodes");

            // Calculate the target network size
            int
                inputs = x.GetLength(1),
                outputs = ys.GetLength(1);
            if (size == null) size = (inputs + outputs) / 2;

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] w1w2)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(inputs, size.Value, outputs, w1w2);
                return network.CalculateCost(x, ys);
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] w1w2)
            {
                NeuralNetwork network = NeuralNetwork.Deserialize(inputs, size.Value, outputs, w1w2);
                return network.CostFunctionPrime(x, ys);
            }

            // Initialize the optimization function
            IAccordNETGradientOptimizationMethod bfgs = AccordNETGradientOptimizationMethodCompatibilityWrapper.Instance.Invoke(
                inputs * size.Value + size.Value * outputs, // Number of free variables in the function to optimize
                CostFunction, GradientFunction);
            bfgs.Token = token;

            // Handle the progress if necessary
            if (progress != null) bfgs.ProgressRelay += (s, e) =>
            {
                progress.Report(new BackpropagationProgressEventArgs(
                    () => NeuralNetwork.Deserialize(inputs, size.Value, outputs, e.Solution), e.Iteration, e.Value));
            };

            // Minimize the cost function
            await Task.Run(() =>
            {
                if (solution != null) bfgs.Minimize(solution);
                else bfgs.Minimize();
            }, token);

            // Return the result network
            return NeuralNetwork.Deserialize(inputs, size.Value, outputs, bfgs.Solution);
        }
    }
}
