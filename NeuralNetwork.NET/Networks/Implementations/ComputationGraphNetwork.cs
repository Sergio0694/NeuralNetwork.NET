using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Graph;
using NeuralNetworkNET.Networks.Graph.Nodes;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A computational graph network with an arbitrary internal structure and number of computation branches
    /// </summary>
    internal sealed class ComputationGraphNetwork : NeuralNetworkBase
    {
        #region Parameters

        /// <inheritdoc/>
        public override ref readonly TensorInfo InputInfo => ref Graph.Root.Children[0].To<IComputationGraphNode, ProcessingNode>().Layer.InputInfo;
        
        /// <inheritdoc/>
        public override ref readonly TensorInfo OutputInfo => ref OutputLayer.OutputInfo;

        /// <inheritdoc/>
        public override IReadOnlyList<INetworkLayer> Layers { get; }

        /// <inheritdoc/>
        protected override OutputLayerBase OutputLayer { get; }

        /// <inheritdoc/>
        public override int Size => Graph.Nodes.Count;

        /// <summary>
        /// The underlying layers graph for the network
        /// </summary>
        [NotNull]
        [JsonProperty(nameof(Graph), Order = 7)]
        private readonly ComputationGraph Graph;
        
        #endregion

        public ComputationGraphNetwork([NotNull] ComputationGraph graph) : base(NetworkType.ComputationGraph)
        {
            Graph = graph;
            OutputLayer = Graph.OutputNode.To<IComputationGraphNode, ProcessingNode>().Layer.To<INetworkLayer, OutputLayerBase>();
            Layers = Graph.ProcessingNodes.Select(node => node.Layer).ToArray();
            WeightedLayersIndexes = Layers.Select((l, i) => (Layer: l as WeightedLayerBase, Index: i)).Where(t => t.Layer != null).Select(t => t.Index).ToArray();
        }

        #region Implementation

        /// <inheritdoc/>
        protected override void Forward(in Tensor x, out Tensor yHat)
        {
            // Local mapping
            using (TensorMap<IComputationGraphNode> aMap = new TensorMap<IComputationGraphNode> { [Graph.Root] = x })
            {
                // Recursive forward function
                void Forward(IComputationGraphNode node)
                {
                    switch (node)
                    {
                        case ProcessingNode processing:
                        {
                            processing.Layer.To<INetworkLayer, NetworkLayerBase>().Forward(aMap[processing.Parent], out Tensor z, out Tensor a);
                            z.Free();
                            aMap[processing] = a;
                            if (processing == Graph.OutputNode) return;
                            break;
                        }
                        case DepthConcatenationNode concatenation:
                            if (!TryExecuteMergeForward(aMap, concatenation, out Tensor m)) return;
                            aMap[concatenation] = m;
                            break;
                        case SumNode sum:
                        {
                            if (!TryExecuteMergeForward(aMap, sum, out Tensor z, out Tensor a)) return;
                            z.Free();
                            aMap[sum] = a;
                            break;
                        }
                        case TrainingNode _: return;
                        default:
                            throw new ArgumentException("The node type is not supported", nameof(node));
                    }
                    for (int i = 0; i < node.Children.Count; i++)
                        if (!aMap.ContainsKey(node.Children[i]))
                            Forward(node.Children[i]);
                }

                // Manually start the forward pass on the first input branches
                for (int i = 0; i < Graph.Root.Children.Count; i++)
                    if (!aMap.ContainsKey(Graph.Root.Children[i]))
                        Forward(Graph.Root.Children[i]);

                // Collect the outputs and return
                aMap.Remove(Graph.Root);
                yHat = aMap[Graph.OutputNode];
                aMap.Remove(Graph.OutputNode); // Remove yHat from the map to keep it allocated
            }
        }

        /// <inheritdoc/>
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")] // Tensor maps in optimization closure
        internal override unsafe void Backpropagate(in SamplesBatch batch, float dropout, WeightsUpdater updater)
        {
            fixed (float* px = batch.X, py = batch.Y)
            {
                Tensor.Reshape(px, batch.X.GetLength(0), batch.X.GetLength(1), out Tensor x);
                Tensor.Reshape(py, batch.Y.GetLength(0), batch.Y.GetLength(1), out Tensor y);

                // Local mapping
                using (TensorMap<IComputationGraphNode> 
                    zMap = new TensorMap<IComputationGraphNode>(),
                    aMap = new TensorMap<IComputationGraphNode> { [Graph.Root] = x }, // Input tensor to remove before disposal
                    dropMap = new TensorMap<IComputationGraphNode>(),
                    dMap = new TensorMap<IComputationGraphNode>(),
                    dJdwMap = new TensorMap<IComputationGraphNode>(),
                    dJdbMap = new TensorMap<IComputationGraphNode>())
                {
                    #region Forward

                    /* =================
                     * Forward pass
                     * =================
                     * Propagate the input tensor through the network, across all the
                     * available branches, and store the computed activities, activations and
                     * dropout masks along the way */
                    void Forward(IComputationGraphNode node)
                    {
                        switch (node)
                        {
                            case ProcessingNode processing:
                            {
                                processing.Layer.To<INetworkLayer, NetworkLayerBase>().Forward(aMap[processing.Parent], out Tensor z, out Tensor a);
                                zMap[processing] = z;
                                aMap[processing] = a;
                                if (processing.Layer.LayerType == LayerType.FullyConnected && dropout > 0)
                                {
                                    ThreadSafeRandom.NextDropoutMask(a.Entities, a.Length, dropout, out Tensor mask);
                                    CpuBlas.MultiplyElementwise(a, mask, a);
                                    dropMap[processing] = mask;
                                }
                                break;
                            }
                            case DepthConcatenationNode concatenation:
                                if (!TryExecuteMergeForward(aMap, concatenation, out Tensor m)) return;
                                aMap[concatenation] = m;
                                break;
                            case SumNode sum:
                            {
                                if (!TryExecuteMergeForward(aMap, sum, out Tensor z, out Tensor a)) return;
                                zMap[sum] = z;
                                aMap[sum] = a;
                                break;
                            }
                            case TrainingNode split: 
                                aMap[split.Parent].Duplicate(out Tensor copy);
                                aMap[node] = copy;
                                break;
                            default:
                                throw new ArgumentException("The node type is not supported", nameof(node));
                        }
                        for (int i = 0; i < node.Children.Count; i++)
                            if (!aMap.ContainsKey(node.Children[i]))
                                Forward(node.Children[i]);
                    }

                    // Manually start the forward pass on the first input branches
                    for (int i = 0; i < Graph.Root.Children.Count; i++)
                        if (!aMap.ContainsKey(Graph.Root.Children[i]))
                            Forward(Graph.Root.Children[i]);

                    #endregion

                    /* ===================
                     * Backpropagation
                     * ===================
                     * Propagate the output error delta through the graph network
                     * and compute the gradients for all the weighted layers in the graph.
                     * For each auxiliary classifier, backpropagate the error delta from that branch
                     * as well, and merge it back into the main graph when the training branch parent
                     * node is reached. This can help reducing the vanishing gradient problem. */
                    void Backward(IComputationGraphNode node)
                    {
                        if (node is InputNode) return;

                        #region Output delta extraction

                        /* ==============
                         * Extract dy
                         * ==============
                         * For each graph node, calculate the output error delta from the previous node
                         * in the backpropagation path. Different cases are present here:
                         * 1:   The current node has a single child node. In this case, either use a shallow
                         *      copy of the previous dy tensor (as processing nodes are guaranteed not to edit
                         *      their dy parameter) or duplicate the previous dy (for paass-through nodes)
                         * 2:   The current node has multiple children. First, check if all the required dy
                         *      tensors are available. If not, stop here and wait for another branch to reach
                         *      this position. If they are available, extract the useful data and sum the
                         *      dy parameters together to be used in the current node.
                         * 3:   The current node is an output layer, so no dy tensor is needed. */
                        Tensor dy;
                        bool linked = false;
                        if (node.Children.Count == 1)
                        {
                            if (node.Children[0] is DepthConcatenationNode merge)
                            {
                                int offset = 0, length = -1;
                                for (int j = 0; j < merge.Parents.Count; j++)
                                {
                                    length = aMap[merge.Parents[j]].Length;
                                    if (merge.Parents[j] == node) break;
                                    offset += j == 0 ? 0 : aMap[merge.Parents[j - 1]].Length;
                                }
                                Tensor.New(x.Entities, length, out dy);
                                CpuDnn.DepthConcatenationBackward(dMap[merge], offset, dy);
                            }
                            else if (node.Type == ComputationGraphNodeType.Processing)
                            {
                                dy = dMap[node.Children[0]];
                                linked = true; // Just use a shallow copy, but mark it as non-disposable
                            }
                            else dMap[node.Children[0]].Duplicate(out dy);
                        }
                        else if (node.Children.Count > 1)
                        {
                            // Ensure the required tensors are present
                            for (int i = 0; i < node.Children.Count; i++)
                                if (!dMap.ContainsKey(node.Children[i]))
                                    return;

                            // Extract the dy tensor
                            Tensor* dyt = stackalloc Tensor[node.Children.Count];
                            bool* links = stackalloc bool[node.Children.Count];
                            for (int i = 0; i < node.Children.Count; i++)
                            {
                                // Extract the tensor slice from a depth concatenation layer
                                if (node.Children[i] is DepthConcatenationNode merge)
                                {
                                    int offset = 0, length = -1;
                                    for (int j = 0; j < merge.Parents.Count; j++)
                                    {
                                        length = aMap[merge.Parents[j]].Length;
                                        if (merge.Parents[j] == node) break;
                                        offset += j == 0 ? 0 : aMap[merge.Parents[j - 1]].Length;
                                    }
                                    Tensor.New(x.Entities, length, out dyt[i]);
                                    CpuDnn.DepthConcatenationBackward(dMap[merge], offset, dyt[i]);
                                }
                                else
                                {
                                    dyt[i] = dMap[node.Children[i]];
                                    links[i] = true;
                                }
                            }

                            // Sum the output deltas
                            Tensor.Like(*dyt, out dy);
                            CpuBlas.Sum(new Span<Tensor>(dyt, node.Children.Count), dy);
                            for (int i = 0; i < node.Children.Count; i++)
                                if (!links[i])
                                    dyt[i].Free();
                        }
                        else dy = Tensor.Null; // The current node is an output node

                        #endregion

                        /* ==================
                         * Node processing
                         * ==================
                         * Execute the backpropagation pass on the current node and calculate the
                         * weights and biases gradient if necessary.
                         * As with all the other implementation methods here, the network graph structure
                         * is assumed to be already verified and valid. */
                        switch (node)
                        {
                            case ProcessingNode processing:

                                // Backpropagation with optional gradient
                                if (processing.Layer is ConstantLayerBase constant)
                                {
                                    if (processing.Parent is InputNode)
                                    {
                                        if (!linked) dMap[node] = dy; // Notify other branches, if present
                                        return; // No backpropagation required
                                    }
                                    Tensor.New(x.Entities, constant.InputInfo.Size, out Tensor dx);
                                    constant.Backpropagate(aMap[processing.Parent], zMap[node], dy, dx);
                                    dMap[node] = dx;
                                }
                                else if (processing.Layer is WeightedLayerBase weighted)
                                {
                                    if (dropMap.TryGetValue(node, out Tensor mask)) CpuBlas.MultiplyElementwise(dy, mask, dy); // Optional dropout
                                    Tensor dJdw, dJdb;
                                    if (processing.Parent is InputNode) weighted.Backpropagate(x, zMap[node], dy, Tensor.Null, out dJdw, out dJdb);
                                    else if (weighted is OutputLayerBase output)
                                    {
                                        Tensor.New(x.Entities, output.InputInfo.Size, out Tensor dx);
                                        output.Backpropagate(aMap[processing.Parent], aMap[node], y, zMap[node], dx, out dJdw, out dJdb);
                                        dMap[node] = dx;
                                    }
                                    else
                                    {
                                        // Gradients and backpropagated error delta
                                        Tensor.New(x.Entities, weighted.InputInfo.Size, out Tensor dx);
                                        weighted.Backpropagate(aMap[processing.Parent], zMap[node], dy, dx, out dJdw, out dJdb);
                                        dMap[node] = dx;
                                    }
                                    dJdwMap[node] = dJdw;
                                    dJdbMap[node] = dJdb;
                                }
                                else throw new InvalidOperationException("Invalid backpropagation node");
                                if (!linked) dy.TryFree();
                                if (!dMap.ContainsKey(processing.Parent))
                                    Backward(processing.Parent);
                                break;
                            case DepthConcatenationNode concatenation:
                                dMap[node] = dy; // Pass-through the error delta with no changes
                                for (int i = 0; i < concatenation.Parents.Count; i++)
                                    if (!dMap.ContainsKey(concatenation.Parents[i]))
                                        Backward(concatenation.Parents[i]);
                                break;
                            case SumNode sum:
                            {
                                Tensor.Like(dy, out Tensor dx); // Inputs and outputs have the same shape for sum nodes
                                sum.Backpropagate(zMap[node], dy, dx);
                                dy.Free();
                                dMap[node] = dx;
                                for (int i = 0; i < sum.Parents.Count; i++)
                                    if (!dMap.ContainsKey(sum.Parents[i]))
                                        Backward(sum.Parents[i]);
                                break;
                            }
                            case TrainingNode split:
                                dMap[node] = dy;
                                Backward(split.Parent); // No check needed for parent dy in map, as this is a training branch
                                break;
                            default:
                                throw new ArgumentException("The node type is not supported", nameof(node));
                        }
                    }

                    // Backpropagate from the training and inference outputs
                    for (int i = 0; i < Graph.TrainingOutputNodes.Count; i++) Backward(Graph.TrainingOutputNodes[i]);
                    Backward(Graph.OutputNode);

                    /* ================
                     * Optimization
                     * ================
                     * Edit the network weights according to the computed gradients */
                    int samples = batch.X.GetLength(0);
                    Parallel.For(0, WeightedLayersIndexes.Length, i =>
                    {
                        ProcessingNode node = Graph.ProcessingNodes[WeightedLayersIndexes[i]];
                        updater(i, dJdwMap[node], dJdbMap[node], samples, node.Layer.To<INetworkLayer, WeightedLayerBase>());
                    }).AssertCompleted();
                    aMap.Remove(Graph.Root); // Remove the input tensor
                }
            }
        }

        // Executes the forward pass on a merge node, if possible
        private static unsafe bool TryExecuteMergeForward([NotNull] TensorMap<IComputationGraphNode> map, [NotNull] SumNode sum, out Tensor z, out Tensor a)
        {
            // Prepare the inputs
            Tensor* xs = stackalloc Tensor[sum.Parents.Count];
            for (int i = 0; i < sum.Parents.Count; i++)
            {
                if (!map.TryGetValue(sum.Parents[i], out xs[i]))
                {
                    z = a = default;
                    return false;
                }
            }

            // Forward through the merge node
            sum.Forward(new Span<Tensor>(xs, sum.Parents.Count), out z, out a);
            return true;
        }

        // Executes the forward pass on a depth concatenation node, if possible
        private static unsafe bool TryExecuteMergeForward([NotNull] TensorMap<IComputationGraphNode> map, [NotNull] DepthConcatenationNode concatenation, out Tensor y)
        {
            // Prepare the inputs
            Tensor* xs = stackalloc Tensor[concatenation.Parents.Count];
            for (int i = 0; i < concatenation.Parents.Count; i++)
            {
                if (!map.TryGetValue(concatenation.Parents[i], out xs[i]))
                {
                    y = default;
                    return false;
                }
            }
            Span<Tensor> inputs = new Span<Tensor>(xs, concatenation.Parents.Count);

            // Forward through the merge node
            int l = 0;
            for (int i = 0; i < inputs.Length; i++) l += inputs[i].Length;
            Tensor.New(xs[0].Entities, l, out y);
            CpuDnn.DepthConcatenationForward(inputs, y);
            return true;
        }

        #endregion

        #region Features extraction

        /// <inheritdoc/>
        public override IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures(float[] x)
        {
            return (from pair in ExtractDeepFeatures(x.AsSpan().AsMatrix(1, x.Length))
                    let z = pair.Z?.Flatten()
                    let a = pair.A.Flatten()
                    select (z, a)).ToArray();
        }

        /// <inheritdoc/>
        [SuppressMessage("ReSharper", "AccessToDisposedClosure")]
        public override unsafe IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures(float[,] x)
        {
            // Local mapping           
            fixed (float* px = x)
            {
                Tensor.Reshape(px, x.GetLength(0), x.GetLength(1), out Tensor xc);
                using (TensorMap<IComputationGraphNode> 
                    zMap = new TensorMap<IComputationGraphNode>(),
                    aMap = new TensorMap<IComputationGraphNode> { [Graph.Root] = xc })
                {
                    void Forward(IComputationGraphNode node)
                    {
                        switch (node)
                        {
                            case ProcessingNode processing:
                            {
                                processing.Layer.To<INetworkLayer, NetworkLayerBase>().Forward(aMap[processing.Parent], out Tensor z, out Tensor a);
                                zMap[processing] = z;
                                aMap[processing] = a;
                                break;
                            }
                            case DepthConcatenationNode concatenation:
                                if (!TryExecuteMergeForward(aMap, concatenation, out Tensor m)) return;
                                aMap[concatenation] = m;
                                break;
                            case SumNode sum:
                            {
                                if (!TryExecuteMergeForward(aMap, sum, out Tensor z, out Tensor a)) return;
                                zMap[sum] = z;
                                aMap[sum] = a;
                                break;
                            }
                            case TrainingNode _: return;
                            default:
                                throw new ArgumentException("The node type is not supported", nameof(node));
                        }          
                        for (int i = 0; i < node.Children.Count; i++)
                            if (!aMap.ContainsKey(node.Children[i]))
                                Forward(node.Children[i]);
                    }

                    // Manually start the forward pass on the first input branches
                    for (int i = 0; i < Graph.Root.Children.Count; i++)
                        if (!aMap.ContainsKey(Graph.Root.Children[i]))
                            Forward(Graph.Root.Children[i]);

                    // Return the extracted features
                    aMap.Remove(Graph.Root);
                    return Graph.Nodes.Where(node => zMap.ContainsKey(node) || aMap.ContainsKey(node)).Select(node =>
                    {
                        aMap.TryGetValue(node, out Tensor at);
                        float[,] 
                            z = zMap.TryGetValue(node, out Tensor zt) ? zt.ToArray2D() : null,
                            a = at.ToArray2D();
                        return (z, a);
                    }).ToArray();
                }
            }
        }

        #endregion

        #region Misc

        /// <inheritdoc/>
        protected override void Serialize(Stream stream)
        {
            stream.Write(InputInfo);
            Graph.Serialize(stream);
        }

        /// <summary>
        /// Tries to deserialize a new <see cref="ComputationGraphNetwork"/> from the input <see cref="Stream"/>
        /// </summary>
        /// <param name="stream">The input <see cref="Stream"/> to use to read the network data</param>
        /// <param name="preference">The layers deserialization preference</param>
        [MustUseReturnValue, CanBeNull]
        public static INeuralNetwork Deserialize([NotNull] Stream stream, ExecutionModePreference preference)
        {
            if (!stream.TryRead(out TensorInfo inputs)) return null;
            Func<TensorInfo, ComputationGraph> f = ComputationGraph.Deserialize(stream, preference);
            return f == null ? null : new ComputationGraphNetwork(f(inputs));
        }

        /// <inheritdoc/>
        public override bool Equals(INeuralNetwork other) => base.Equals(other) && other is ComputationGraphNetwork network && Graph.Equals(network.Graph);

        /// <inheritdoc/>
        public override INeuralNetwork Clone() => new ComputationGraphNetwork(Graph.GetCloneFactory()(InputInfo));

        #endregion
    }
}
