using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using JetBrains.Annotations;
using ConvolutionalNeuralNetworkLibrary.Convolution;

namespace ConvolutionalNeuralNetworkLibrary
{
    public class NetworkTrainer
    {
        private readonly NeuralNetwork InitialNetwork;

        private readonly Func<double[], double> CostFunction;

        /// <summary>
        /// Initializes a new instance for the input network
        /// </summary>
        /// <param name="network">The neural network to train</param>
        private NetworkTrainer([NotNull] NeuralNetwork network,
            [NotNull] Func<double[], double> costFunction)
        {
            InitialNetwork = network;
            CostFunction = costFunction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="pipeline"></param>
        /// <param name="ys"></param>
        /// <param name="size"></param>
        [PublicAPI]
        [Pure]
        [NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        public static NeuralNetwork New(
            [NotNull] IReadOnlyList<double[,]> data, 
            [NotNull] ConvolutionPipeline pipeline, 
            [NotNull] double[,] ys, int size)
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

            // Function to reconstruct a network from the input serialized weights
            NeuralNetwork DeserializeNetwork(double[] w1w2)
            {
                // Reconstruct the matrices for the network
                double[,]
                    w1 = new double[inputs, size],
                    w2 = new double[size, outputs];
                int w1length = sizeof(double) * w1.Length;
                Buffer.BlockCopy(w1w2, 0, w1, 0, w1length);
                Buffer.BlockCopy(w1w2, w1length, w2, 0, sizeof(double) * w2.Length);

                // Create the new network to use
                return new NeuralNetwork(inputs, outputs, size, w1, w2);
            }

            // Calculates the cost for a network with the input weights
            double CostFunction(double[] w1w2)
            {
                NeuralNetwork network = DeserializeNetwork(w1w2);
                return network.CalculateCost(x, ys);
            }

            // Calculates the gradient for a network with the input weights
            double[] GradientFunction(double[] w1w2)
            {
                NeuralNetwork network = DeserializeNetwork(w1w2);
                (double[,] dJdW1, double[,] dJdW2) gradient = network.CostFunctionPrime(x, ys);
                return gradient.dJdW1.Cast<double>().Concat(gradient.dJdW2.Cast<double>()).ToArray();
            }

            // TODO
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(
                inputs * size + size * outputs, // Number of free variables in the function to optimize
                CostFunction, GradientFunction);

            // TODO
            bfgs.Progress += (s, e) => System.Diagnostics.Debug.WriteLine($"# {e.Iteration} ---> {e.Value}");
            bfgs.Minimize();
            return null;
        }
    }
}
