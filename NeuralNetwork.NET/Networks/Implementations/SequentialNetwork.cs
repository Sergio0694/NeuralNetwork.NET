using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.Networks.Layers.Cpu;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A complete and fully connected neural network with an arbitrary number of hidden layers
    /// </summary>
    internal sealed class SequentialNetwork : NeuralNetworkBase
    {
        #region Base class

        /// <inheritdoc/>
        public override ref readonly TensorInfo InputInfo => ref Layers[0].InputInfo;

        /// <inheritdoc/>
        public override ref readonly TensorInfo OutputInfo => ref Layers[Layers.Count - 1].OutputInfo;

        /// <inheritdoc/>
        public override IReadOnlyList<INetworkLayer> Layers => _Layers;

        #endregion

        /// <summary>
        /// The list of layers that make up the neural network
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly NetworkLayerBase[] _Layers;

        /// <summary>
        /// Initializes a new network with the given parameters
        /// </summary>
        /// <param name="layers">The layers that make up the neural network</param>
        internal SequentialNetwork([NotNull, ItemNotNull] params INetworkLayer[] layers) : base(NetworkType.Sequential)
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
        public override unsafe IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures(float[] x)
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
        public override unsafe IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures(float[,] x)
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

        /// <inheritdoc/>
        protected override void Forward(in Tensor x, out Tensor yHat)
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

        /// <inheritdoc/>
        protected override float CalculateCost(in Tensor x, in Tensor y)
        {
            Forward(x, out Tensor yHat);
            float cost = _Layers[_Layers.Length - 1].To<NetworkLayerBase, OutputLayerBase>().CalculateCost(yHat, y);
            yHat.Free();
            return cost;
        }

        /// <inheritdoc/>
        internal override unsafe void Backpropagate(in SamplesBatch batch, float dropout, [NotNull] WeightsUpdater updater)
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
                        CpuBlas.MultiplyElementwise(aList[i], dropoutMasks[i], aList[i]);
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
                    _Layers[l + 1].Backpropagate(aList[l], *deltas[l + 1], zList[l], _Layers[l].ActivationFunctions.ActivationPrime);
                    if (!dropoutMasks[l].IsNull) CpuBlas.MultiplyElementwise(zList[l], dropoutMasks[l], zList[l]);
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
                    dropoutMasks[i].TryFree();
                }
                zList[_Layers.Length - 1].Free();
                aList[_Layers.Length - 1].Free();
            }
        }

        #endregion

        #region Deserialization and misc

        /// <summary>
        /// Tries to deserialize a new <see cref="SequentialNetwork"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the network data</param>
        /// <param name="preference">The layers deserialization preference</param>
        [MustUseReturnValue, CanBeNull]
        public static INeuralNetwork Deserialize([NotNull] Stream stream, LayersLoadingPreference preference)
        {
            List<INetworkLayer> layers = new List<INetworkLayer>();
            while (stream.TryRead(out LayerType type))
            {
                // Deserialization attempt
                INetworkLayer layer = null;
                if (preference == LayersLoadingPreference.Cuda) layer = NetworkLoader.CuDnnLayerDeserialize(stream, type);
                if (layer == null) layer = NetworkLoader.CpuLayerDeserialize(stream, type);
                if (layer == null) return null;

                // Add to the queue
                layers.Add(layer);
            }
            return new SequentialNetwork(layers.ToArray());
        }

        /// <inheritdoc/>
        public override INeuralNetwork Clone() => new SequentialNetwork(_Layers.Select(l => l.Clone()).ToArray());

        #endregion
    }
}
