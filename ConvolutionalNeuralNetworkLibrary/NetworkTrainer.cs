using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Accord.Math.Optimization;
using JetBrains.Annotations;
using ConvolutionalNeuralNetworkLibrary.Convolution;

namespace ConvolutionalNeuralNetworkLibrary
{
    /// <summary>
    /// A static class that create and trains a neural network for the input data and expected results
    /// </summary>
    public static class NetworkTrainer
    {
        /// <summary>
        /// Generates and trains a neural network suited for the input data and results
        /// </summary>
        /// <param name="data">The raw input data for the supervised training</param>
        /// <param name="pipeline">The convolution pipeline to apply to the input data</param>
        /// <param name="ys">The results vector</param>
        /// <param name="size">The number of nodes in the hidden layer of the network (it will be decided automatically if null)</param>
        /// <param name="token">The cancellation token for the training session</param>
        /// <param name="solution">An optional starting solution to resume a previous training session</param>
        /// <param name="progress">An optional progress callback</param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static NeuralNetwork ComputeTrainedNetwork(
            [NotNull] IReadOnlyList<double[,]> data, 
            [NotNull] ConvolutionPipeline pipeline, 
            [NotNull] double[,] ys, [CanBeNull] int? size,
            CancellationToken token,
            [CanBeNull] double[] solution = null,
            [CanBeNull] IProgress<CNNOptimizationProgress> progress = null)
        {
            // Preliminary checks
            if (data.Count == 0) throw new ArgumentOutOfRangeException("The input set is empty");
            if (ys.Length == 0) throw new ArgumentOutOfRangeException("The results set is empty");
            if (data.Count != ys.GetLength(0)) throw new ArgumentOutOfRangeException("The number of inputs and results must be equal");
            if (size <= 0) throw new ArgumentOutOfRangeException("The hidden layer must have a positive number of nodes");

            // Process the data through the convolution pipeline
            IReadOnlyList<double[][,]> convolutions = pipeline.Process(data);

            // Prepare the base network and the input data
            int 
                depth = convolutions[0].Length, // Depth of each convolution volume
                lsize = convolutions[0][0].Length, // Size of each 2D layer
                ch = convolutions[0][0].GetLength(0), // Height of each convolution layer
                cw = convolutions[0][0].GetLength(1), // Width of each convolution layer
                volume = depth * lsize;
            double[,] x = new double[convolutions.Count, volume]; // Matrix with all the batched inputs
            for (int i = 0; i < convolutions.Count; i++) // Iterate over all the volumes
                for (int j = 0; j < depth; j++) // Iterate over all the depth layer in each volume
                    for (int z = 0; z < ch; z++) // Height of each layer
                        for (int w = 0; w < cw; w++) // Width of each layer
                            x[i, j * lsize + z * ch + w] = convolutions[i][j][z, w];

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
                (double[,] dJdW1, double[,] dJdW2) gradient = network.CostFunctionPrime(x, ys);
                return gradient.dJdW1.Cast<double>().Concat(gradient.dJdW2.Cast<double>()).ToArray();
            }

            // Initialize the optimization function
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(
                inputs * size.Value + size.Value * outputs, // Number of free variables in the function to optimize
                CostFunction, GradientFunction)
            {
                Token = token
            };

            // Handle the progress if necessary
            if (progress != null) bfgs.Progress += (s, e) =>
            {
                progress.Report(new CNNOptimizationProgress((inputs, size.Value, outputs, e.Solution), e.Iteration, e.Value));
            };

            // Minimize the cost function
            if (solution != null) bfgs.Minimize(solution);
            else bfgs.Minimize();

            // Return the result network
            return NeuralNetwork.Deserialize(inputs, size.Value, outputs, bfgs.Solution);
        }
    }
}
