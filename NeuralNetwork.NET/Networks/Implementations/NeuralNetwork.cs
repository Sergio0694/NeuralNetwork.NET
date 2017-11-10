using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Misc;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A complete and fully connected neural network with an arbitrary number of hidden layers
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    public sealed class NeuralNetwork : INeuralNetwork
    {
        #region Public parameters

        /// <inheritdoc/>
        [JsonProperty(nameof(InputLayerSize), Required = Required.Always)]
        public int InputLayerSize => Weights[0].GetLength(0);

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputLayerSize), Required = Required.Always)]
        public int OutputLayerSize => Weights[Weights.Count - 1].GetLength(1);

        private int[] _HiddenLayers;

        /// <inheritdoc/>
        [JsonProperty(nameof(HiddenLayers), Required = Required.Always)]
        public IReadOnlyList<int> HiddenLayers => _HiddenLayers ?? (_HiddenLayers = Weights.Take(Weights.Count - 1).Select(w => w.GetLength(1)).ToArray());

        /// <inheritdoc/>
        [JsonProperty(nameof(ActivationFunctions), Required = Required.Always)]
        public IReadOnlyList<ActivationFunctionType> ActivationFunctions { get; }

        #endregion

        #region Local fields

        /// <summary>
        /// The list of weight matrices for the network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Weights), Required = Required.Always)]
        private readonly IReadOnlyList<double[,]> Weights;

        /// <summary>
        /// The precalculated list of transposed weight matrices to use inthe gradient function
        /// </summary>
        /// <remarks>The first item is always null (to save space), as it isn't needed to calculate the gradient</remarks>
        [NotNull, ItemCanBeNull]
        private readonly double[][,] TransposedWeights;

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
        /// <param name="activations">The activation functions to use in the new network</param>
        public NeuralNetwork([NotNull] IReadOnlyList<double[,]> weights, [NotNull] IReadOnlyList<double[]> biases, [NotNull] IReadOnlyList<ActivationFunctionType> activations)
        {
            // Input check
            if (weights.Count == 0) throw new ArgumentOutOfRangeException(nameof(weights), "The weights must have a length at least equal to 1");
            if (activations.Count != weights.Count) throw new ArgumentOutOfRangeException(nameof(activations), "The number of activations must be the same as the weights");
            if (biases.Count != weights.Count) throw new ArgumentException(nameof(biases), "The bias vector has an invalid size");
            for (int i = 0; i < weights.Count; i++)
            {
                if (i > 0 && weights[i - 1].GetLength(1) != weights[i].GetLength(0))
                    throw new ArgumentOutOfRangeException(nameof(weights), "Some weight matrix doesn't have the right size");
                if (activations[i] != ActivationFunctionType.Sigmoid && activations[i] != ActivationFunctionType.Tanh && i < weights.Count - 1)
                    throw new ArgumentOutOfRangeException(nameof(activations), $"The {activations[i]} activation function can only be used in the output layer");
                if (weights[i].GetLength(1) != biases[i].Length)
                    throw new ArgumentException(nameof(biases), $"The bias vector #{i} doesn't have the right size");
            }

            // Parameters setup
            Weights = weights;
            Biases = biases;
            ActivationFunctions = activations;
            TransposedWeights = new double[weights.Count][,];
        }

        /// <summary>
        /// Creates a new random instance with the given number of neurons in each layer
        /// </summary>
        /// <param name="layers">The type of layers that make up the network</param>
        [NotNull]
        public static NeuralNetwork NewRandom([NotNull, ItemNotNull] params NetworkLayer[] layers)
        {
            // Check
            if (layers.Length < 2) throw new ArgumentOutOfRangeException(nameof(layers), "The network must have at least two layers");
            if (!(layers[0] is NetworkLayer.InputLayer)) throw new ArgumentException(nameof(layers), "The first layer isn't a valid input layer");

            // Initialize the weights
            Random random = new Random();
            double[][,] weights = new double[layers.Length - 1][,];
            double[][] biases = new double[layers.Length - 1][];
            ActivationFunctionType[] activations = new ActivationFunctionType[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                int fanIn = layers[i].Neurons, fanOut = layers[i + 1].Neurons;
                if (!(layers[i + 1] is NetworkLayer.FullyConnectedLayer fullyConnected))
                    throw new ArgumentException(nameof(layers), $"The layer #{i + 1} isn't a valid fully connected layer");
                activations[i] = fullyConnected.Activation;
                switch (fullyConnected.Activation)
                {
                    case ActivationFunctionType.Sigmoid:
                        weights[i] = random.NextSigmoidMatrix(fanIn, fanOut);
                        break;
                    case ActivationFunctionType.Tanh:
                        weights[i] = random.NextTanhMatrix(fanIn, fanOut);
                        break;
                    default:
                        weights[i] = random.NextXavierMatrix(fanIn, fanOut);
                        break;
                }
                biases[i] = random.NextGaussianVector(fanOut);
            }
            return new NeuralNetwork(weights, biases, activations);
        }

        #endregion

        #region Single processing

        /// <inheritdoc/>
        public double[] Forward(double[] x) => Forward(x.ToMatrix()).Flatten();

        /// <inheritdoc/>
        public double CalculateCost(double[] x, double[] y) => CalculateCost(x.ToMatrix(), y.ToMatrix());

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="y">The expected result</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal IReadOnlyList<LayerGradient> Backpropagate([NotNull] double[] x, [NotNull] double[] y) => Backpropagate(x.ToMatrix(), y.ToMatrix());

        #endregion

        #region Batch processing

        /// <inheritdoc/>
        public double[,] Forward(double[,] x)
        {
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // A(l) = activation(W(l) * A(l - 1) + b(l))
                ActivationFunction activation = ActivationFunctionProvider.GetActivation(ActivationFunctions[i]);
                a0 = MatrixServiceProvider.MultiplyWithSumAndActivation(a0, Weights[i], Biases[i], activation);
            }
            return a0; // At least one weight matrix, so a0 != x
        }

        /// <inheritdoc/>
        public double CalculateCost(double[,] input, double[,] y)
        {
            // Forward the input
            double[,] yHat = Forward(input);

            // Calculate the cost (half the squared difference)
            return MatrixServiceProvider.HalfSquaredDifference(yHat, y);
        }

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="y">The expected results</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal IReadOnlyList<LayerGradient> Backpropagate([NotNull] double[,] x, [NotNull] double[,] y)
        {
            // Feedforward
            int steps = Weights.Count;  // Number of forward hops through the network
            double[][,]
                zList = new double[steps][,],
                aList = new double[steps][,];
            ActivationFunction[] activationPrimes = new ActivationFunction[Weights.Count];
            double[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Save the intermediate steps to be able to reuse them later
                double[,] zi = MatrixServiceProvider.MultiplyWithSum(a0, Weights[i], Biases[i]);
                zList[i] = zi;
                ActivationFunctionType type = ActivationFunctions[i];
                activationPrimes[i] = ActivationFunctionProvider.GetActivationPrime(type);
                ActivationFunction activation = ActivationFunctionProvider.GetActivation(type);
                aList[i] = a0 = MatrixServiceProvider.Activation(zi, activation);
            }

            /* ============================
             * Calculate delta(L) in place
             * ============================
             * Perform the sigmoid prime of zL, the activity on the last layer
             * Calculate the gradient of C with respect to a, so (yHat - y)
             * Compute d(L), the Hadamard product of the gradient and the sigmoid prime for L */
            double[,] dL = aList[aList.Length - 1];
            MatrixServiceProvider.InPlaceSubtractAndHadamardProductWithActivationPrime(dL, y, zList[zList.Length - 1], activationPrimes[activationPrimes.Length - 1]);

            // Backpropagation
            double[][,] deltas = new double[steps][,];      // One additional delta for each hop, delta(L) has already been calculated
            deltas[steps - 1] = dL;                         // Store the delta(L) in the last position
            for (int l = Weights.Count - 2; l >= 0; l--)    // Loop for l = L - 1, L - 2, ..., 2
            {
                // Prepare d(l + 1) and W(l + 1)T
                double[,]
                    transposed = TransposedWeights[l + 1] ?? (TransposedWeights[l + 1] = Weights[l + 1].Transpose()), // Calculate W[l + 1]T if needed
                    dl = zList[l]; // Local reference on the delta to calculate in place

                /* ============================
                 * Calculate delta(l) in place
                 * ============================
                 * Perform the sigmoid prime of z(l), the activity on the previous layer
                 * Multiply the previous delta with the transposed weights of the following layer
                 * Compute d(l), the Hadamard product of z'(l) and delta(l + 1) * W(l + 1)T */
                MatrixServiceProvider.MultiplyAndInPlaceActivationPrimeAndHadamardProduct(dl, deltas[l + 1], transposed, activationPrimes[l]);
                deltas[l] = dl;
            }

            // Compute the gradient
            int dLength = Weights.Sum(w => w.Length) + deltas.Sum(d => d.GetLength(1));
            LayerGradient[] gradient = new LayerGradient[dLength]; // One gradient item for layer
            for (int i = 0; i < Weights.Count; i++)
            {
                // Store the target delta
                double[,] di = deltas[i];

                // Compute dJdw(l) and dJdb(l)
                double[,] dJdw = i == 0
                    ? MatrixServiceProvider.TransposeAndMultiply(x, di)             // dJdW1, transposed input * first delta
                    : MatrixServiceProvider.TransposeAndMultiply(aList[i - 1], di); // dJdWi, previous activation transposed * current delta
                double[] dJdb = di.CompressVertically();
                gradient[i] = new LayerGradient(dJdw, dJdb);
            }
            return gradient;
        }

        #endregion

        #region Tools

        /// <summary>
        /// Deserializes a neural network from the input weights and parameters
        /// </summary>
        /// <param name="data">The data representing the weights of the network</param>
        /// <param name="layers">The list of network layers</param>
        [PublicAPI]
        [Pure, NotNull]
        internal static NeuralNetwork Deserialize([NotNull] double[] data, [NotNull, ItemNotNull] params NetworkLayer[] layers)
        {
            // Checks
            if (layers.Length < 2) throw new ArgumentException("The network must have at least 2 layers");
            if (!(layers[0] is NetworkLayer.InputLayer)) throw new ArgumentException(nameof(layers), "The first layer isn't a valid input layer");

            // Parse the input data
            int depth = layers.Length - 1;
            double[][,] weights = new double[depth][,];
            double[][] biases = new double[depth][];
            ActivationFunctionType[] activations = new ActivationFunctionType[weights.Length];
            int position = 0;
            for (int i = 0; i < depth; i++)
            {
                // Unpack the current weights
                int fanIn = layers[i].Neurons, fanOut = layers[i + 1].Neurons;
                double[,] wi = new double[fanIn, fanOut];
                int bytes = sizeof(double) * wi.Length;
                Buffer.BlockCopy(data, position, wi, 0, bytes);
                position += bytes;
                weights[i] = wi;
                if (!(layers[i + 1] is NetworkLayer.FullyConnectedLayer fullyConnected))
                    throw new ArgumentException(nameof(layers), $"The layer #{i + 1} isn't a valid fully connected layer");
                activations[i] = fullyConnected.Activation;

                // Unpack the current bias vector
                double[] bias = new double[fanOut];
                bytes = sizeof(double) * bias.Length;
                Buffer.BlockCopy(data, position, bias, 0, bytes);
                position += bytes;
                biases[i] = bias;
            }
            if (position / sizeof(double) != data.Length) throw new InvalidOperationException("Invalid network requested size");

            // Create the new network to use
            return new NeuralNetwork(weights, biases, activations);
        }

        /// <summary>
        /// Serializes the current network into a binary representation
        /// </summary>
        /// <returns>A <see cref="double"/> array containing all the weights and biases of the network</returns>
        [PublicAPI]
        [Pure]
        internal double[] Serialize()
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

        /// <inheritdoc/>
        public String SerializeAsJSON() => JsonConvert.SerializeObject(this, Formatting.Indented);

        // Creates a new instance from another network with the same structure
        [Pure, NotNull]
        internal NeuralNetwork Crossover([NotNull] NeuralNetwork other, [NotNull] Random random)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public bool Equals(INeuralNetwork other)
        {
            // Compare general features
            if (other is NeuralNetwork network &&
                other.InputLayerSize == InputLayerSize &&
                other.OutputLayerSize == OutputLayerSize &&
                other.HiddenLayers.SequenceEqual(HiddenLayers) &&
                other.ActivationFunctions.SequenceEqual(ActivationFunctions))
            {
                // Compare each weight and bias value
                for (int i = 0; i < Weights.Count; i++)
                    if (!network.Weights[i].ContentEquals(Weights[i]) ||
                        !network.Biases[i].ContentEquals(Biases[i])) return false;
                return true;
            }
            return false;
        }

        #endregion

        public void StochasticGradientDescent(
            (double[,] X, double[,] Y) trainingSet,
            (double[,] X, double[,] Y) testSet,
            int epochs, int batchSize, double eta)
        {
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            while (epochs-- > 0)
            {
                double cost = CalculateCost(testSet.X, testSet.Y);
                Console.WriteLine($"Cost: {cost}");
                double[,] yHat = Forward(testSet.X);
                int total = 0;
                for (int i = 0; i < yHat.GetLength(0); i++)
                {
                    double[]
                        yHatSample = new double[10],
                        ySample = new double[10];
                    Buffer.BlockCopy(yHat, sizeof(double) * i * 10, yHatSample, 0, sizeof(double) * 10);
                    Buffer.BlockCopy(testSet.Y, sizeof(double) * i * 10, ySample, 0, sizeof(double) * 10);
                    int
                        maxHat = yHatSample.IndexOfMax(),
                        max = ySample.IndexOfMax();
                    if (max == maxHat) total++;
                }
                Console.WriteLine($"{total} / {testSet.Y.GetLength(0)}");

                foreach (TrainingBatch batch in batches.NextEpoch())
                {
                    IReadOnlyList<LayerGradient> dJ = Backpropagate(batch.X, batch.Y);
                    bool loopResult = Parallel.For(0, Weights.Count, i =>
                    {
                        for (int x = 0; x < Weights[i].GetLength(0); x++)
                        for (int y = 0; y < Weights[i].GetLength(1); y++)
                        {
                            Weights[i][x, y] -= eta / batch.X.GetLength(0) * dJ[i].DJdw[x, y];
                        }

                        for (int x = 0; x < Biases[i].Length; x++)
                        {
                            Biases[i][x] -= eta / batch.X.GetLength(0) * dJ[i].Djdb[x];
                        }
                    }).IsCompleted;
                    if (!loopResult) throw new InvalidOperationException("Error performing the parallel loop");
                }
            }
        }
    }
}
