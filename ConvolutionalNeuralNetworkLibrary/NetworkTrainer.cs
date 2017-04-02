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
        public NetworkTrainer([NotNull] NeuralNetwork network,
            [NotNull] Func<double[], double> costFunction)
        {
            InitialNetwork = network;
            CostFunction = costFunction;
        }

        public async void Foo(IReadOnlyList<double[,]> data, ConvolutionPipeline pipeline, IReadOnlyList<double[]> ys)
        {
            // Process the data through the convolution pipeline
            IReadOnlyList<double[][,]> convolutions = await Task.Run(() => pipeline.Process(data));

            NeuralNetwork DeserializeNetwork(double[] w1w2)
            {
                // Reconstruct the matrices for the network
                double[,]
                    w1 = new double[InitialNetwork.InputLayerSize, InitialNetwork.HiddenLayerSize],
                    w2 = new double[InitialNetwork.HiddenLayerSize, InitialNetwork.OutputLayerSize];
                int w1length = sizeof(double) * w1.Length;
                Buffer.BlockCopy(w1w2, 0, w1, 0, w1length);
                Buffer.BlockCopy(w1w2, w1length, w2, 0, sizeof(double) * w2.Length);

                // Create the new network to use
                NeuralNetwork network = new NeuralNetwork(
                    InitialNetwork.InputLayerSize, InitialNetwork.OutputLayerSize,
                    InitialNetwork.HiddenLayerSize, w1, w2);
                return network;
            }

            double CostFunction(double[] w1w2)
            {
                NeuralNetwork network = DeserializeNetwork(w1w2);
                double cost = 0;
                for (int i = 0; i < convolutions.Count; i++)
                {
                    double[] serialized = convolutions[i].SelectMany(c => c.Cast<double>()).ToArray();
                    cost += network.CalculateCost(serialized, ys[i]);
                }
                return cost;
            }

            double[] GradientFunction(double[] w1w2)
            {
                NeuralNetwork network = DeserializeNetwork(w1w2);
                return null; // TODO
            }

            // TODO
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(
                InitialNetwork.InputLayerSize * InitialNetwork.HiddenLayerSize + InitialNetwork.HiddenLayerSize * InitialNetwork.OutputLayerSize,
                CostFunction, GradientFunction);       
        }
    }
}
