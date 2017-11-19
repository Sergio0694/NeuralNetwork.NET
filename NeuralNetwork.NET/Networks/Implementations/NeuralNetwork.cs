using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Exceptions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.Networks.Implementations.Layers.APIs;
using NeuralNetworkNET.Networks.Implementations.Misc;
using NeuralNetworkNET.Networks.PublicAPIs;
using NeuralNetworkNET.SupervisedLearning.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Misc;
using NeuralNetworkNET.SupervisedLearning.Optimization.Parameters;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

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
        [JsonProperty(nameof(InputLayerSize), Required = Required.Always, Order = 1)]
        public int InputLayerSize { get; }

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputLayerSize), Required = Required.Always, Order = 2)]
        public int OutputLayerSize { get; }

        #endregion

        /// <summary>
        /// The list of layers that make up the neural network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Layers), Required = Required.Always, Order = 3)]
        private readonly IReadOnlyList<NetworkLayerBase> Layers;

        /// <summary>
        /// Initializes a new network with the given parameters
        /// </summary>
        /// <param name="layers">The layers that make up the neural network</param>
        internal NeuralNetwork([NotNull, ItemNotNull] params INetworkLayer[] layers)
        {
            // Input check
            if (layers.Length < 1) throw new ArgumentOutOfRangeException(nameof(layers), "The network must have at least one layer");
            foreach ((NetworkLayerBase layer, int i) in layers.Select((l, i) => (l as NetworkLayerBase, i)))
            {
                if (i != layers.Length - 1 && layer is OutputLayerBase) throw new ArgumentException("The output layer must be the last layer in the network");
                if (i == layers.Length - 1 && !(layer is OutputLayerBase)) throw new ArgumentException("The last layer must be an output layer");
                if (i > 0 && layers[i - 1].Outputs != layer.Inputs) throw new ArgumentException($"The inputs of layer #{i} don't match with the outputs of the previous layer");
            }

            // Parameters setup
            InputLayerSize = layers[0].Inputs;
            OutputLayerSize = layers[layers.Length - 1].Outputs;
            Layers = layers.Cast<NetworkLayerBase>().ToArray();
        }

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
            float[,] yHat = x;
            foreach (NetworkLayerBase layer in Layers)
                (_, yHat) = layer.Forward(yHat); // Forward the inputs through all the network layers
            return yHat;
        }

        /// <inheritdoc/>
        public float CalculateCost(float[,] input, float[,] y)
        {
            // Forward the input
            float[,] yHat = Forward(input);

            // Calculate the cost
            return Layers[Layers.Count - 1].To<NetworkLayerBase, OutputLayerBase>().CalculateCost(yHat, y);
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
            float[][,]
                zList = new float[Layers.Count][,],
                aList = new float[Layers.Count][,];
            foreach ((NetworkLayerBase layer, int i) in Layers.Select((l, i) => (l, i)))
            {
                // Save the intermediate steps to be able to reuse them later
                (zList[i], aList[i]) = layer.Forward(i == 0 ? x : aList[i - 1]);
            }

            // Backpropagation deltas
            float[][,] deltas = new float[Layers.Count][,]; // One delta for each hop through the network

            /* ======================
             * Calculate delta(L)
             * ======================
             * Perform the sigmoid prime of zL, the activity on the last layer
             * Calculate the gradient of C with respect to a
             * Compute d(L), the Hadamard product of the gradient and the sigmoid prime for L.
             * NOTE: for some cost functions (eg. log-likelyhood) the sigmoid prime and the Hadamard product
             *       with the first part of the formula are skipped as that factor is simplified during the calculation of the output delta */
            deltas[deltas.Length - 1] = Layers[Layers.Count - 1].To<NetworkLayerBase, OutputLayerBase>().Backpropagate(aList[aList.Length -1], y, zList[zList.Length - 1]);
            for (int l = Layers.Count - 2; l >= 0; l--)
            {
                /* ======================
                 * Calculate delta(l)
                 * ======================
                 * Perform the sigmoid prime of z(l), the activity on the previous layer
                 * Multiply the previous delta with the transposed weights of the following layer
                 * Compute d(l), the Hadamard product of z'(l) and delta(l + 1) * W(l + 1)T */
                deltas[l] = Layers[l + 1].Backpropagate(deltas[l + 1], zList[l], Layers[l].ActivationFunctions.ActivationPrime);
            }

            /* =============================================
             * Compute the gradients DJDw(l) and DJDb(l)
             * =============================================
             * Compute the gradients for each layer with weights and biases.
             * NOTE: the gradient is only computed for layers with weights and biases, for all the other
             *       layers a dummy gradient is added to the list and then ignored during the weights update pass */
            LayerGradient[] gradient = new LayerGradient[Layers.Count]; // One gradient item for layer
            foreach ((WeightedLayerBase layer, int i) in Layers.Select((l, i) => (l as WeightedLayerBase, i)))
            {
                gradient[i] = layer.ComputeGradient(i == 0 ? x : aList[i - 1], deltas[i]);
            }
            return gradient;
        }

        #endregion

        #region Training

        /// <summary>
        /// Trains the current network using the gradient descent algorithm
        /// </summary>
        /// <param name="trainingSet">The training set for the current session</param>
        /// <param name="epochs">The desired number of training epochs to run</param>
        /// <param name="batchSize">The size of each training batch</param>
        /// <param name="validationParameters">The optional <see cref="ValidationParameters"/> instance (used for early-stopping)</param>
        /// <param name="testParameters">The optional <see cref="TestParameters"/> instance (used to monitor the training progress)</param>
        /// <param name="eta">The learning rate</param>
        /// <param name="lambda">The L2 regularization factor</param>
        /// <param name="token">The <see cref="CancellationToken"/> for the training session</param>
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
            float alpha = eta / batchSize;
            IEnumerable<(WeightedLayerBase Layer, LayerGradient Gradient)> targets =
                from layer in Layers.Select((l, i) => (Layer: l as WeightedLayerBase, Index: i))
                where layer.Layer != null
                select (layer.Layer, dJ[layer.Index]);
            Parallel.ForEach(targets, target => target.Layer.Minimize(target.Gradient, alpha, l2Factor));
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
                cost = Layers[Layers.Count - 1].To<NetworkLayerBase, OutputLayerBase>().CalculateCost(yHat, evaluationSet.Y),
                accuracy = (float)total / h * 100;
            return (cost, total, accuracy);
        }

        #endregion

        #region Tools

        /// <inheritdoc/>
        public String SerializeAsJSON() => JsonConvert.SerializeObject(this, Formatting.Indented, new StringEnumConverter());

        /// <inheritdoc/>
        public bool Equals(INeuralNetwork other)
        {
            // Compare general features
            if (other is NeuralNetwork network &&
                other.InputLayerSize == InputLayerSize &&
                other.OutputLayerSize == OutputLayerSize)
            {
                // Compare the individual layers
                for (int i = 0; i < Layers.Count; i++)
                    if (!Layers[i].Equals(network.Layers[i])) return false;
                return true;
            }
            return false;
        }

        #endregion
    }
}
