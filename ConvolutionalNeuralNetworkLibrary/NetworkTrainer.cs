using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary
{
    public class NetworkTrainer
    {
        private readonly NeuralNetwork Network;

        private readonly Func<double[], double> CostFunction;

        /// <summary>
        /// Initializes a new instance for the input network
        /// </summary>
        /// <param name="network">The neural network to train</param>
        public NetworkTrainer([NotNull] NeuralNetwork network,
            [NotNull] Func<double[], double> costFunction)
        {
            Network = network;
            CostFunction = costFunction;
        }

        public void Foo()
        {
            // TODO
            BoundedBroydenFletcherGoldfarbShanno bfgs = new BoundedBroydenFletcherGoldfarbShanno(1);       
        }
    }
}
