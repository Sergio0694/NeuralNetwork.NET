using JetBrains.Annotations;

namespace ConvolutionalNeuralNetworkLibrary
{
    /// <summary>
    /// A structure that contains the progress data while optimizing a network
    /// </summary>
    public class CNNOptimizationProgress
    {
        // Private parameters for the lazy network initialization
        private readonly (int In, int Size, int Out, double[] Weights) Parameters;

        private NeuralNetwork _Network;

        /// <summary>
        /// Gets the current neural network for the reached optimization progress
        /// </summary>
        [NotNull]
        public NeuralNetwork Network => _Network 
            ?? (_Network = NeuralNetwork.Deserialize(Parameters.In, Parameters.Size, Parameters.Out, Parameters.Weights));

        /// <summary>
        /// Gets the current iteration number
        /// </summary>
        public int Iteration { get; }

        /// <summary>
        /// Gets the current value for the function to optimize
        /// </summary>
        public double Cost { get; }

        // Internal constructor
        internal CNNOptimizationProgress((int, int, int, double[]) parameters, int iteration, double cost)
        {
            Parameters = parameters;
            Iteration = iteration;
            Cost = cost;
        }
    }
}
