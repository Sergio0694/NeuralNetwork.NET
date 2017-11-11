using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Implementations.Misc;
using NeuralNetworkNET.Networks.Layers;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
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

        /// <inheritdoc/>
        [JsonProperty(nameof(ActivationFunctions), Required = Required.Always)]
        public CostFunctionType CostFunction { get; }

        #endregion

        #region Local fields

        /// <summary>
        /// The list of weight matrices for the network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Weights), Required = Required.Always)]
        private readonly IReadOnlyList<float[,]> Weights;

        /// <summary>
        /// The precalculated list of transposed weight matrices to use inthe gradient function
        /// </summary>
        /// <remarks>The first item is always null (to save space), as it isn't needed to calculate the gradient</remarks>
        [NotNull, ItemCanBeNull]
        private readonly float[][,] TransposedWeights;

        /// <summary>
        /// The list of bias vectors for the network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Biases), Required = Required.Always)]
        private readonly IReadOnlyList<float[]> Biases;

        #endregion

        #region Initialization

        /// <summary>
        /// Initializes a new instance with the given parameters
        /// </summary>
        /// <param name="weights">The weights in all the network layers</param>
        /// <param name="biases">The bias vectors to use in the network</param>
        /// <param name="activations">The activation functions to use in the new network</param>
        /// <param name="costFunction">The desired cost function to use</param>
        public NeuralNetwork(
            [NotNull] IReadOnlyList<float[,]> weights, 
            [NotNull] IReadOnlyList<float[]> biases, 
            [NotNull] IReadOnlyList<ActivationFunctionType> activations, CostFunctionType costFunction)
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
            TransposedWeights = new float[weights.Count][,];
            CostFunction = costFunction;
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
            float[][,] weights = new float[layers.Length - 1][,];
            float[][] biases = new float[layers.Length - 1][];
            ActivationFunctionType[] activations = new ActivationFunctionType[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                int fanIn = layers[i].Neurons, fanOut = layers[i + 1].Neurons;
                if (!(layers[i + 1] is NetworkLayer.FullyConnectedLayer fullyConnected))
                    throw new ArgumentException(nameof(layers), $"The layer #{i + 1} isn't a valid fully connected layer");
                activations[i] = fullyConnected.Activation;
                weights[i] = random.NextXavierMatrix(fanIn, fanOut);
                biases[i] = random.NextGaussianVector(fanOut);
            }
            if (!(layers[layers.Length - 1] is NetworkLayer.OutputLayer output))
                throw new ArgumentException(nameof(layers), "The last layer isn't a valid output layer");
            CostFunctionType costFunction = output.Cost;
            return new NeuralNetwork(weights, biases, activations, costFunction);
        }

        #endregion

        #region Single processing

        /// <inheritdoc/>
        public float[] Forward(float[] x) => Forward(x.ToMatrix()).Flatten();

        /// <inheritdoc/>
        public float CalculateCost(float[] x, float[] y) => CalculateCost(x.ToMatrix(), y.ToMatrix());

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="y">The expected result</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal IReadOnlyList<LayerGradient> Backpropagate([NotNull] float[] x, [NotNull] float[] y) => Backpropagate(x.ToMatrix(), y.ToMatrix());

        #endregion

        #region Batch processing

        /// <inheritdoc/>
        public float[,] Forward(float[,] x)
        {
            float[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // A(l) = activation(W(l) * A(l - 1) + b(l))
                ActivationFunction activation = ActivationFunctionProvider.GetActivation(ActivationFunctions[i]);
                a0 = MatrixServiceProvider.MultiplyWithSumAndActivation(a0, Weights[i], Biases[i], activation);
            }
            return a0; // At least one weight matrix, so a0 != x
        }

        /// <inheritdoc/>
        public float CalculateCost(float[,] input, float[,] y)
        {
            // Forward the input
            float[,] yHat = Forward(input);

            // Calculate the cost (half the squared difference)
            return yHat.CrossEntropyCost(y);
        }

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="x">The input data</param>
        /// <param name="y">The expected results</param>
        [PublicAPI]
        [Pure, NotNull]
        [CollectionAccess(CollectionAccessType.Read)]
        internal IReadOnlyList<LayerGradient> Backpropagate([NotNull] float[,] x, [NotNull] float[,] y)
        {
            // Feedforward
            int steps = Weights.Count;  // Number of forward hops through the network
            float[][,]
                zList = new float[steps][,],
                aList = new float[steps][,];
            ActivationFunction[] activationPrimes = new ActivationFunction[Weights.Count];
            float[,] a0 = x;
            for (int i = 0; i < Weights.Count; i++)
            {
                // Save the intermediate steps to be able to reuse them later
                float[,] zi = MatrixServiceProvider.MultiplyWithSum(a0, Weights[i], Biases[i]);
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
            float[,] dL = aList[aList.Length - 1];
            CostFunctionProvider.GetCostFunctionPrime(CostFunction).Invoke(dL, y, zList[zList.Length - 1], activationPrimes[activationPrimes.Length - 1]);

            // Backpropagation
            float[][,] deltas = new float[steps][,];      // One additional delta for each hop, delta(L) has already been calculated
            deltas[steps - 1] = dL;                         // Store the delta(L) in the last position
            for (int l = Weights.Count - 2; l >= 0; l--)    // Loop for l = L - 1, L - 2, ..., 2
            {
                // Prepare d(l + 1) and W(l + 1)T
                float[,]
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
            LayerGradient[] gradient = new LayerGradient[Weights.Count]; // One gradient item for layer
            for (int i = 0; i < Weights.Count; i++)
            {
                // Store the target delta
                float[,] di = deltas[i];

                // Compute dJdw(l) and dJdb(l)
                float[,] dJdw = i == 0
                    ? MatrixServiceProvider.TransposeAndMultiply(x, di)             // dJdW1, transposed input * first delta
                    : MatrixServiceProvider.TransposeAndMultiply(aList[i - 1], di); // dJdWi, previous activation transposed * current delta
                float[] dJdb = di.CompressVertically();
                gradient[i] = new LayerGradient(dJdw, dJdb);
            }
            return gradient;
        }

        #endregion

        #region Training

        public TrainingStopReason StochasticGradientDescent(
            (float[,] X, float[,] Y) trainingSet,
            int epochs, int batchSize,
            ValidationParameters validationParameters = null,
            TestParameters testParameters = null,
            float eta = 0.5f, float lambda = 0.1f,
            CancellationToken token = default)
        {
            // Convergence manager for the validation dataset
            RelativeConvergence convergence = validationParameters == null
                ? null
                : new RelativeConvergence(validationParameters.Tolerance, validationParameters.EpochsInterval);

            // Create the training batches
            int trainingSamples = trainingSet.X.GetLength(0);
            float l2Factor = eta * lambda / trainingSamples;
            BatchesCollection batches = BatchesCollection.FromDataset(trainingSet, batchSize);
            for (int i = 0; i < epochs; i++)
            {
                // Gradient descent over the current batches
                foreach (TrainingBatch batch in batches.NextEpoch())
                {
                    if (token.IsCancellationRequested) return TrainingStopReason.TrainingCanceled;
                    IReadOnlyList<LayerGradient> dJ = Backpropagate(batch.X, batch.Y);
                    int size = batch.X.GetLength(0);
                    UpdateWeights(dJ, size, eta, l2Factor);
                }

                // Check the validation dataset
                if (convergence != null)
                {
                    (_, _, float accuracy) = Evaluate(validationParameters.Dataset);
                    convergence.Value = accuracy;
                    if (convergence.HasConverged) return TrainingStopReason.EarlyStopping;
                }

                // Report progress if necessary
                if (testParameters != null)
                {
                    (float cost, _, float accuracy) = Evaluate(testParameters.Dataset);
                    testParameters.ProgressCallback.Report(new BackpropagationProgressEventArgs(i + 1, cost, accuracy));
                }
            }
            return TrainingStopReason.EpochsCompleted;
        }

        /// <summary>
        /// Updates the current network weights after a backpropagation on a training batch
        /// </summary>
        /// <param name="dJ">The gradient for the cost function over the last training batch</param>
        /// <param name="batchSize">The size of the last training batch</param>
        /// <param name="eta">The learning rate for the training session</param>
        /// <param name="l2Factor">The L2 regularization factor</param>
        private void UpdateWeights([NotNull] IReadOnlyList<LayerGradient> dJ, int batchSize, float eta, float l2Factor)
        {
            // Divide the workload
            int blocks = Weights.Count * 2;
            float scale = eta / batchSize;

            // Function to update the weights from the calculated gradient
            unsafe void Kernel(int i)
            {
                // Get the index of the current layer and branch over weights/biases
                int l = i / 2;
                if (i % 2 == 0)
                {
                    // Tweak the weights of the lth layer
                    float[,] weight = Weights[l];
                    fixed (float* pw = weight, pdj = dJ[l].DJdw)
                    {
                        int
                            h = weight.GetLength(0),
                            w = weight.GetLength(1);
                        for (int x = 0; x < h; x++)
                        {
                            int offset = x * w;
                            for (int y = 0; y < w; y++)
                            {
                                int target = offset + y;
                                pw[target] -= l2Factor * pw[target] + scale * pdj[target];
                            }
                        }
                    }
                }
                else
                {
                    // Tweak the biases of the lth layer
                    float[] bias = Biases[l];
                    fixed (float* pb = bias, pdj = dJ[l].Djdb)
                    {
                        int w = bias.Length;
                        for (int x = 0; x < w; x++)
                            pb[x] -= scale * pdj[x];
                    }
                }
            }
            Parallel.For(0, blocks, Kernel).AssertCompleted();
        }

        /// <summary>
        /// Calculates the current network performances with the given test samples
        /// </summary>
        /// <param name="evaluationSet">The inputs and expected outputs to test the network</param>
        internal (float Cost, int Classified, float Accuracy) Evaluate((float[,] X, float[,] Y) evaluationSet)
        {
            // Feedforward
            float[,] yHat = Forward(evaluationSet.X);
            int
                h = evaluationSet.X.GetLength(0),
                wy = evaluationSet.Y.GetLength(1),
                total = 0;

            // Function that counts the correctly classified items
            unsafe void Kernel(int i)
            {
                fixed (float* pyHat = yHat, pY = evaluationSet.Y)
                {
                    int
                        offset = i * wy,
                        maxHat = MatrixExtensions.Argmax(pyHat + offset, wy),
                        max = MatrixExtensions.Argmax(pY + offset, wy);
                    if (max == maxHat) Interlocked.Increment(ref total);
                }
            }

            // Check the correctly classified samples and calculate the cost
            Parallel.For(0, h, Kernel).AssertCompleted();
            float
                cost = CostFunctionProvider.GetCostFunction(CostFunction).Invoke(yHat, evaluationSet.Y),
                accuracy = (float)total / h * 100;
            return (cost, total, accuracy);
        }

        #endregion

        #region Tools

        /// <inheritdoc/>
        public String SerializeAsJSON() => JsonConvert.SerializeObject(this, Formatting.Indented); // TODO: check and test

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
    }
}
