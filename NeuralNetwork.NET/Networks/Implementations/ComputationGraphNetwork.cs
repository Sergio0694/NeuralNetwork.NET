using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Helpers;
using NeuralNetworkNET.Networks.Graph;
using NeuralNetworkNET.Networks.Layers.Abstract;
using NeuralNetworkNET.SupervisedLearning.Data;
using NeuralNetworkNET.SupervisedLearning.Optimization;

namespace NeuralNetworkNET.Networks.Implementations
{
    /// <summary>
    /// A computational graph network with an arbitrary internal structure and number of computation branches
    /// </summary>
    internal sealed class ComputationGraphNetwork : NeuralNetworkBase
    {
        /// <inheritdoc/>
        public override ref readonly TensorInfo InputInfo => ref Graph.Root.Children[0].To<IComputationGraphNode, ProcessingNode>().Layer.InputInfo;
        
        /// <inheritdoc/>
        public override ref readonly TensorInfo OutputInfo => ref OutputLayer.OutputInfo;

        /// <inheritdoc/>
        public override IReadOnlyList<INetworkLayer> Layers { get; }

        /// <inheritdoc/>
        protected override OutputLayerBase OutputLayer { get; }

        /// <summary>
        /// The underlying layers graph for the network
        /// </summary>
        [NotNull]
        private readonly ComputationGraph Graph;

        public ComputationGraphNetwork([NotNull] IComputationGraphNode root) : base(NetworkType.ComputationGraph)
        {
            Graph = new ComputationGraph(root);
            OutputLayer = Graph.OutputNode.To<IComputationGraphNode, ProcessingNode>().Layer.To<INetworkLayer, OutputLayerBase>();
            Layers = Graph.Layers.Select(node => node.To<IComputationGraphNode, ProcessingNode>().Layer).ToArray();
        }

        public override IReadOnlyList<(float[] Z, float[] A)> ExtractDeepFeatures(float[] x)
        {
            throw new NotImplementedException();
        }

        public override IReadOnlyList<(float[,] Z, float[,] A)> ExtractDeepFeatures(float[,] x)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        protected override unsafe void Forward(in Tensor x, out Tensor yHat)
        {
            // Local mapping
            Dictionary<IComputationGraphNode, int> mergeMap = new Dictionary<IComputationGraphNode, int>();
            using (TensorMap<IComputationGraphNode> 
                zMap = new TensorMap<IComputationGraphNode>(),
                aMap = new TensorMap<IComputationGraphNode>())
            {
                // Recursive forward function
                void Forward(IComputationGraphNode node)
                {
                    switch (node)
                    {
                        case ProcessingNode processing:
                            processing.Layer.To<INetworkLayer, NetworkLayerBase>().Forward(aMap[processing.Parent], out Tensor z, out Tensor a);
                            zMap[processing] = z;
                            aMap[processing] = a;
                            if (processing == Graph.OutputNode) return;
                            foreach (IComputationGraphNode child in processing.Children)
                                Forward(child);
                            break;
                        case MergeNode merge:
                            if (mergeMap.TryGetValue(merge, out int value) && value == merge.Parents.Count - 1)
                            {
                                // Prepare the inputs
                                Tensor* xs = stackalloc Tensor[merge.Parents.Count];
                                for (int i = 0; i < merge.Parents.Count; i++)
                                    xs[i] = aMap[merge.Parents[i]];
                                Span<Tensor> inputs = new Span<Tensor>(xs, merge.Parents.Count);

                                // Forward through the merge node
                                Tensor.New(xs[0].Entities, xs[0].Length, out Tensor y);
                                if (merge.Type == ComputationGraphNodeType.Sum) CpuDnn.SumForward(inputs, y);
                                else if (merge.Type == ComputationGraphNodeType.DepthStacking) CpuDnn.DepthConcatenationForward(inputs, y);
                                else throw new ArgumentOutOfRangeException(nameof(merge.Type), "Unsupported node type");
                                aMap[merge] = y;
                            }
                            else mergeMap[merge]++;
                            break;
                        case TrainingSplitNode split:
                            Forward(split.InferenceBranchNode);
                            break;
                        default:
                            throw new ArgumentException("The node type is not supported", nameof(node));
                    }
                }

                // Manually start the forward pass on the first input branches
                foreach (IComputationGraphNode child in Graph.Root.Children)
                {
                    child.To<IComputationGraphNode, ProcessingNode>().Layer.To<INetworkLayer, NetworkLayerBase>().Forward(x, out Tensor z, out Tensor a);
                    zMap[child] = z;
                    aMap[child] = a;
                    Forward(child);
                }

                // Collect the outputs and return
                yHat = aMap[Graph.OutputNode];
                aMap.Remove(Graph.OutputNode); // Remove yHat from the map to keep it allocated
            }
        }

        internal override void Backpropagate(in SamplesBatch batch, float dropout, WeightsUpdater updater)
        {
            throw new NotImplementedException();
        }

        public override INeuralNetwork Clone()
        {
            throw new NotImplementedException();
        }
    }
}
