using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    public sealed class ComputationGraph
    {
        /// <summary>
        /// Gets the root <see cref="IComputationGraphNode"/> for the current graph
        /// </summary>
        [NotNull]
        public IComputationGraphNode Root { get; }

        /// <summary>
        /// Gets the in-order serialized view of the network graph nodes
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<IComputationGraphNode> Layers;

        /// <summary>
        /// Gets the graph main output node
        /// </summary>
        [NotNull]
        internal readonly IComputationGraphNode OutputNode;

        /// <summary>
        /// Gets the training output nodes, if present
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyCollection<IComputationGraphNode> TrainingOutputNodes;
        
        internal ComputationGraph(IComputationGraphNode root)
        {
            Root = root;
            (Layers, OutputNode, TrainingOutputNodes) = ExtractGraphInfo(root);
        }

        #region Tools

        /// <summary>
        /// Extracts the info on the input computation graph, and validates its structure
        /// </summary>
        /// <param name="root">The root <see cref="IComputationGraphNode"/> for the computation graph</param>
        [Pure]
        private static (IReadOnlyList<IComputationGraphNode> Nodes, IComputationGraphNode output, IReadOnlyCollection<IComputationGraphNode> trainingOutputs) ExtractGraphInfo(IComputationGraphNode root)
        {
            // Exploration setup
            HashSet<IComputationGraphNode> nodes = new HashSet<IComputationGraphNode>();
            Dictionary<Guid, IComputationGraphNode> trainingOutputs = new Dictionary<Guid, IComputationGraphNode>();
            IComputationGraphNode output = null;

            // Function to recursively explore and validate the graph
            bool Explore(IComputationGraphNode node, Guid trainingId)
            {
                // Add the current node, if not existing
                if (nodes.Contains(node)) return false;
                nodes.Add(node);

                // Explore the graph
                switch (node)
                {
                    case ProcessingNode processing:
                        if (processing.Layer is OutputLayerBase)
                        {
                            if (processing.Children.Count > 0) throw new ArgumentException("An output node can't have any child nodes");
                            if (trainingId != default)
                            {
                                if (trainingOutputs.ContainsKey(trainingId))
                                    throw new ArgumentException("A training branch can only have a single output node");
                                trainingOutputs.Add(trainingId, node);
                            }
                            else if (output == null) output = node;
                            else throw new ArgumentException("The graph can only have a single inference output node");
                        }
                        else
                        {
                            if (processing.Children.Count == 0) throw new ArgumentException("A processing node can't have 0 child nodes");
                            foreach (IComputationGraphNode child in processing.Children)
                                if (!Explore(child, trainingId))
                                    return false;
                        }
                        break;
                    case MergeNode merge:
                        foreach (IComputationGraphNode child in merge.Children)
                            if (!Explore(child, trainingId))
                                return false;
                        break;
                    case TrainingSplitNode split:
                        if (trainingId != default) throw new ArgumentException("A training branch can't contain training split nodes");
                        if (split.InferenceBranchNode.Type == ComputationGraphNodeType.TrainingSplit)
                            throw new ArgumentException("The inference branch of a training split node can't start with another training split node");
                        if (!Explore(split.InferenceBranchNode, default)) return false;
                        if (!Explore(split.TrainingBranchNode, Guid.NewGuid())) return false;
                        break;
                    default: throw new ArgumentException("Invalid node type", nameof(node));
                }
                return true;
            }

            // Return the graph info
            if (!Explore(root, default) || output == null) throw new ArgumentException("The input network doesn't have a valid structure");
            return (nodes.ToArray(), output, trainingOutputs.Values);
        }

        #endregion
    }
}
