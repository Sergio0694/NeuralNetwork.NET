using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Implementations.Layers;
using NeuralNetworkNET.Networks.Implementations.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Data;
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
        [JsonProperty(nameof(InputInfo), Order = 1)]
        public TensorInfo InputInfo => Layers[0].InputInfo;

        /// <inheritdoc/>
        [JsonProperty(nameof(OutputInfo), Order = 2)]
        public TensorInfo OutputInfo => Layers[Layers.Count - 1].OutputInfo;

        /// <inheritdoc/>
        public IReadOnlyList<INetworkLayer> Layers => _Layers;

        #endregion

        /// <summary>
        /// The list of layers that make up the neural network
        /// </summary>
        [NotNull, ItemNotNull]
        [JsonProperty(nameof(Layers), Order = 3)]
        internal readonly NetworkLayerBase[] _Layers;

        // The list of layers with weights to update
        internal readonly int[] WeightedLayersIndexes;

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
                if (i > 0 && layers[i - 1].OutputInfo.Size != layer.InputInfo.Size)
                    throw new ArgumentException($"The inputs of layer #{i} don't match with the outputs of the previous layer");
                if (i > 0 && layer is PoolingLayer && 
                    layers[i - 1] is ConvolutionalLayer convolutional && convolutional.ActivationFunctionType != ActivationFunctionType.Identity)
                    throw new ArgumentException("A convolutional layer followed by a pooling layer must use the Identity activation function");
            }

            // Parameters setup
            _Layers = layers.Cast<NetworkLayerBase>().ToArray();
            WeightedLayersIndexes = layers.Select((l, i) => (Layer: l as WeightedLayerBase, Index: i)).Where(t => t.Layer != null).Select(t => t.Index).ToArray();
        }

        #region Public APIs

        /// <inheritdoc/>
        public unsafe float[] Forward(float[] x)
        {
            fixed (float* px = x)
            {
                Tensor.Reshape(px, 1, x.Length, out Tensor xTensor);
                Forward(xTensor, out Tensor yHatTensor);
                float[] yHat = yHatTensor.ToArray();
                yHatTensor.Free();
                return yHat;
            }
        }

        /// <inheritdoc/>
        public unsafe float CalculateCost(float[] x, float[] y)
        {
            fixed (float* px = x, py = y)
            {
                Tensor.Reshape(px, 1, x.Length, out Tensor xTensor);
                Tensor.Reshape(py, 1, y.Length, out Tensor yTensor);
                return CalculateCost(xTensor, yTensor);
            }
        }

        /// <inheritdoc/>
        public unsafe IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures(float[] x)
        {
            fixed (float* px = x)
            {
                Tensor.Reshape(px, 1, x.Length, out Tensor xTensor);
                Tensor*
                    zList = stackalloc Tensor[_Layers.Length],
                    aList = stackalloc Tensor[_Layers.Length];
                for (int i = 0; i < _Layers.Length; i++)
                {
                    _Layers[i].Forward(i == 0 ? xTensor : aList[i - 1], out zList[i], out aList[i]);
                }
                (float[], float[])[] features = new(float[], float[])[_Layers.Length];
                for (int i = 0; i < _Layers.Length; i++)
                {
                    features[i] = (zList[i].ToArray(), aList[i].ToArray());
                    zList[i].Free();
                    aList[i].Free();
                }
                return features;
            }
        }

        /// <inheritdoc/>
        public unsafe float[,] Forward(float[,] x)
        {
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                Forward(xTensor, out Tensor yHatTensor);
                float[,] yHat = yHatTensor.ToArray2D();
                yHatTensor.Free();
                return yHat;
            }
        }

        /// <inheritdoc/>
        public unsafe float CalculateCost(float[,] x, float[,] y)
        {
            fixed (float* px = x, py = y)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                Tensor.Reshape(py, y.GetLength(0), y.GetLength(1), out Tensor yTensor);
                return CalculateCost(xTensor, yTensor);
            }
        }

        /// <inheritdoc/>
        public unsafe IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures(float[,] x)
        {
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xTensor);
                Tensor*
                    zList = stackalloc Tensor[_Layers.Length],
                    aList = stackalloc Tensor[_Layers.Length];
                for (int i = 0; i < _Layers.Length; i++)
                {
                    _Layers[i].Forward(i == 0 ? xTensor : aList[i - 1], out zList[i], out aList[i]);
                }
                (float[,], float[,])[] features = new(float[,], float[,])[_Layers.Length];
                for (int i = 0; i < _Layers.Length; i++)
                {
                    features[i] = (zList[i].ToArray2D(), aList[i].ToArray2D());
                    zList[i].Free();
                    aList[i].Free();
                }
                return features;
            }
        }

        #endregion

        #region Implementation

        private void Forward(in Tensor x, out Tensor yHat)
        {
            Tensor input = x;
            for (int i = 0; i < _Layers.Length; i++)
            {
                _Layers[i].Forward(input, out Tensor z, out Tensor a); // Forward the inputs through all the network layers
                z.Free();
                if (i > 0) input.Free();
                input = a;
            }
            yHat = input;
        }

        private float CalculateCost(in Tensor x, in Tensor y)
        {
            Forward(x, out Tensor yHat);
            float cost = _Layers[_Layers.Length - 1].To<NetworkLayerBase, OutputLayerBase>().CalculateCost(yHat, y);
            yHat.Free();
            return cost;
        }

        /// <summary>
        /// Calculates the gradient of the cost function with respect to the individual weights and biases
        /// </summary>
        /// <param name="batch">The input training batch</param>
        /// <param name="dropout">The dropout probability for eaach neuron in a <see cref="LayerType.FullyConnected"/> layer</param>
        /// <param name="updater">The function to use to update the network weights after calculating the gradient</param>
        internal unsafe void Backpropagate(in TrainingBatch batch, float dropout, [NotNull] WeightsUpdater updater)
        {
            fixed (float* px = batch.X, py = batch.Y)
            {
                // Setup
                Tensor*
                    zList = stackalloc Tensor[_Layers.Length],
                    aList = stackalloc Tensor[_Layers.Length],
                    dropoutMasks = stackalloc Tensor[_Layers.Length - 1];
                Tensor.Reshape(px, batch.X.GetLength(0), batch.X.GetLength(1), out Tensor x);
                Tensor.Reshape(py, batch.Y.GetLength(0), batch.Y.GetLength(1), out Tensor y);
                Tensor** deltas = stackalloc Tensor*[_Layers.Length]; // One delta for each hop through the network

                // Feedforward
                for (int i = 0; i < _Layers.Length; i++)
                {
                    _Layers[i].Forward(i == 0 ? x : aList[i - 1], out zList[i], out aList[i]);
                    if (_Layers[i].LayerType == LayerType.FullyConnected && dropout > 0)
                    {
                        ThreadSafeRandom.NextDropoutMask(aList[i].Entities, aList[i].Length, dropout, out dropoutMasks[i]);
                        aList[i].InPlaceHadamardProduct(dropoutMasks[i]);
                    }
                }

                /* ======================
                 * Calculate delta(L)
                 * ======================
                 * Perform the sigmoid prime of zL, the activity on the last layer
                 * Calculate the gradient of C with respect to a
                 * Compute d(L), the Hadamard product of the gradient and the sigmoid prime for L.
                 * NOTE: for some cost functions (eg. log-likelyhood) the sigmoid prime and the Hadamard product
                 *       with the first part of the formula are skipped as that factor is simplified during the calculation of the output delta */
                _Layers[_Layers.Length - 1].To<NetworkLayerBase, OutputLayerBase>().Backpropagate(aList[_Layers.Length - 1], y, zList[_Layers.Length - 1]);
                deltas[_Layers.Length - 1] = aList + _Layers.Length - 1;
                for (int l = _Layers.Length - 2; l >= 0; l--)
                {
                    /* ======================
                     * Calculate delta(l)
                     * ======================
                     * Perform the sigmoid prime of z(l), the activity on the previous layer
                     * Multiply the previous delta with the transposed weights of the following layer
                     * Compute d(l), the Hadamard product of z'(l) and delta(l + 1) * W(l + 1)T */
                    _Layers[l + 1].Backpropagate(*deltas[l + 1], zList[l], _Layers[l].ActivationFunctions.ActivationPrime);
                    if (dropoutMasks[l].Ptr != IntPtr.Zero) zList[l].InPlaceHadamardProduct(dropoutMasks[l]);
                    deltas[l] = zList + l;
                }

                /* =============================================
                 * Compute the gradients DJDw(l) and DJDb(l)
                 * =============================================
                 * Compute the gradients for each layer with weights and biases.
                 * NOTE: the gradient is only computed for layers with weights and biases, for all the other
                 *       layers a dummy gradient is added to the list and then ignored during the weights update pass */
                Tensor*
                    dJdw = stackalloc Tensor[WeightedLayersIndexes.Length], // One gradient item for layer
                    dJdb = stackalloc Tensor[WeightedLayersIndexes.Length];
                for (int j = 0; j < WeightedLayersIndexes.Length; j++)
                {
                    int i = WeightedLayersIndexes[j];
                    _Layers[i].To<NetworkLayerBase, WeightedLayerBase>().ComputeGradient(i == 0 ? x : aList[i - 1], *deltas[i], out dJdw[j], out dJdb[j]);
                }

                /* ====================
                 * Gradient descent
                 * ====================
                 * Edit the network weights according to the computed gradients and the current training parameters */
                int samples = batch.X.GetLength(0);
                Parallel.For(0, WeightedLayersIndexes.Length, i =>
                {
                    int l = WeightedLayersIndexes[i];
                    updater(i, dJdw[i], dJdb[i], samples, _Layers[l].To<NetworkLayerBase, WeightedLayerBase>());
                    dJdw[i].Free();
                    dJdb[i].Free();
                }).AssertCompleted();

                // Cleanup
                for (int i = 0; i < _Layers.Length - 1; i++)
                {
                    zList[i].Free();
                    aList[i].Free();
                    if (dropoutMasks[i].Ptr != IntPtr.Zero) dropoutMasks[i].Free();
                }
                zList[_Layers.Length - 1].Free();
                aList[_Layers.Length - 1].Free();
            }
        }

        #endregion

        #region Evaluation

        // Auxiliary function to forward a test batch
        private unsafe (float Cost, int Classified) Evaluate(in Tensor x, in Tensor y)
        {
            // Feedforward
            Forward(x, out Tensor yHat);

            // Function that counts the correctly classified items
            float* pyHat = yHat, pY = y;
            int wy = y.Length, total = 0;
            void Kernel(int i)
            {
                int
                    offset = i * wy,
                    maxHat = MatrixExtensions.Argmax(pyHat + offset, wy),
                    max = MatrixExtensions.Argmax(pY + offset, wy);
                if (max == maxHat) Interlocked.Increment(ref total);
            }

            // Check the correctly classified samples and calculate the cost
            Parallel.For(0, x.Entities, Kernel).AssertCompleted();
            float cost = _Layers[_Layers.Length - 1].To<NetworkLayerBase, OutputLayerBase>().CalculateCost(yHat, y);
            yHat.Free();
            return (cost, total);
        }

        /// <summary>
        /// Calculates the current network performances with the given test samples
        /// </summary>
        /// <param name="evaluationSet">The inputs and expected outputs to test the network</param>
        /// <param name="batchSize">The number of test samples to forward in parallel</param>
        internal unsafe (float Cost, int Classified, float Accuracy) Evaluate((float[,] X, float[,] Y) evaluationSet)
        {
            // Actual test evaluation
            int batchSize = NetworkManager.MaximumBatchSize;
            fixed (float* px = evaluationSet.X, py = evaluationSet.Y)
            {
                int
                    h = evaluationSet.X.GetLength(0),
                    wx = evaluationSet.X.GetLength(1),
                    wy = evaluationSet.Y.GetLength(1),
                    batches = h / batchSize,
                    batchMod = h % batchSize,
                    classified = 0;
                float cost = 0;

                // Process the even batches
                for (int i = 0; i < batches; i++)
                {
                    Tensor.Reshape(px + i * batchSize * wx, batchSize, wx, out Tensor xTensor);
                    Tensor.Reshape(py + i * batchSize * wy, batchSize, wy, out Tensor yTensor);
                    (float pCost, int pClassified) = Evaluate(xTensor, yTensor);
                    cost += pCost;
                    classified += pClassified;
                }

                // Process the remaining samples, if any
                if (batchMod > 0)
                {
                    Tensor.Reshape(px + batches * batchSize * wx, batchMod, wx, out Tensor xTensor);
                    Tensor.Reshape(py + batches * batchSize * wy, batchMod, wy, out Tensor yTensor);
                    (float pCost, int pClassified) = Evaluate(xTensor, yTensor);
                    cost += pCost;
                    classified += pClassified;
                }
                return (cost, classified, (float)classified / h * 100);
            }
        }

        /// <summary>
        /// Calculates the current network performances with the given test samples
        /// </summary>
        /// <param name="batchSize">The training batches currently used to train the network</param>
        internal unsafe (float Cost, int Classified, float Accuracy) Evaluate([NotNull] BatchesCollection batches)
        {
            // Actual test evaluation
            int
                batchSize = NetworkManager.MaximumBatchSize,
                classified = 0;
            float cost = 0;
            for (int i = 0; i < batches.Count; i++)
            {
                ref readonly TrainingBatch batch = ref batches.Batches[i];
                fixed (float* px = batch.X, py = batch.Y)
                {
                    Tensor.Reshape(px, batch.X.GetLength(0), batch.X.GetLength(1), out Tensor xTensor);
                    Tensor.Reshape(py, xTensor.Entities, batch.Y.GetLength(1), out Tensor yTensor);
                    var partial = Evaluate(xTensor, yTensor);
                    cost += partial.Cost;
                    classified += partial.Classified;
                }
            }
            return (cost, classified, (float)classified / batches.Samples * 100);
        }

        #endregion

        #region Serialization and misc

        /// <inheritdoc/>
        public String SerializeMetadataAsJson() => JsonConvert.SerializeObject(this, Formatting.Indented, new StringEnumConverter());

        /// <inheritdoc/>
        public bool Equals(INeuralNetwork other)
        {
            // Compare general features
            if (other is NeuralNetwork network &&
                other.InputInfo == InputInfo &&
                other.OutputInfo == OutputInfo &&
                _Layers.Length == network._Layers.Length)
            {
                // Compare the individual layers
                return _Layers.Zip(network._Layers, (l1, l2) => l1.Equals(l2)).All(b => b);
            }
            return false;
        }

        /// <inheritdoc/>
        public void Save(FileInfo file)
        {
            using (FileStream stream = file.OpenWrite()) 
                Save(stream);
        }

        /// <inheritdoc/>
        public void Save(Stream stream, bool leaveOpen = false)
        {
            using (GZipStream gzip = new GZipStream(stream, CompressionLevel.Optimal, leaveOpen))
                foreach (NetworkLayerBase layer in _Layers) 
                    layer.Serialize(gzip);
        }

        /// <inheritdoc/>
        public INeuralNetwork Clone() => new NeuralNetwork(_Layers.Select(l => l.Clone()).ToArray());

        #endregion
    }
}
