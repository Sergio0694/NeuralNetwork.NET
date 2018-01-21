using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Layers.Abstract;

namespace NeuralNetworkNET.Networks.Graph
{
    /// <summary>
    /// A graph of <see cref="INetworkLayer"/> instances, with O(1) pre-order access time for nodes
    /// </summary>
    internal sealed class ComputationGraph
    {
        /// <summary>
        /// Gets the root <see cref="InputNode"/> for the current graph
        /// </summary>
        [NotNull]
        public InputNode Root { get; }

        /// <summary>
        /// Gets the in-order serialized view of the network graph nodes
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<IComputationGraphNode> Nodes;

        /// <summary>
        /// Gets the in-order serialized view of the <see cref="ProcessingNode"/> instances in the current graph
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyList<ProcessingNode> ProcessingNodes;

        /// <summary>
        /// Gets the graph main output node
        /// </summary>
        [NotNull]
        internal readonly ProcessingNode OutputNode;

        /// <summary>
        /// Gets the training output nodes, if present
        /// </summary>
        [NotNull, ItemNotNull]
        internal readonly IReadOnlyCollection<ProcessingNode> TrainingOutputNodes;
        
        internal ComputationGraph(IComputationGraphNode root)
        {
            Root = root is InputNode input ? input : throw new ArgumentException("The root node isn't valid");
            (Nodes, OutputNode, TrainingOutputNodes) = ExtractGraphInfo(Root);
            ProcessingNodes = Nodes.Pick<IComputationGraphNode, ProcessingNode>().ToArray();
        }

        #region Tools

        /// <summary>
        /// Builds a computation graph from the input node and <see cref="TensorInfo"/> shape
        /// </summary>
        /// <param name="input">The shape of the inputs for the graph to create</param>
        /// <param name="root">The <see cref="NodeBuilder"/> instance that represents the graph root node</param>
        [Pure, NotNull]
        public static IComputationGraphNode BuildGraph(TensorInfo input, [NotNull] NodeBuilder root)
        {
            // Captured dictionary to keep track of graph nodes and tensor shape for each builder
            Dictionary<NodeBuilder, (NodeBase Node, TensorInfo Info)> map = new Dictionary<NodeBuilder, (NodeBase, TensorInfo)>();

            // Function to build the computation graph with a top-down direction
            void BuildMap(NodeBuilder node)
            {
                if (map.ContainsKey(node)) return;
                switch (node.NodeType)
                {
                    case ComputationGraphNodeType.Input:
                        map[node] = (new InputNode(), input);
                        break;
                    case ComputationGraphNodeType.Processing:
                        if (node.Factory == null) throw new InvalidOperationException("Missing layer factory");
                        INetworkLayer layer = node.Factory(map[node.Parents[0]].Info);
                        map[node] = (new ProcessingNode(layer, map[node.Parents[0]].Node), layer.OutputInfo);
                        break;
                    case ComputationGraphNodeType.TrainingBranch:
                        map[node] = (new TrainingNode(map[node.Parents[0]].Node), map[node.Parents[0]].Info);
                        break;
                    case ComputationGraphNodeType.DepthConcatenation:
                    case ComputationGraphNodeType.Sum:

                        // Gather the parent nodes
                        (NodeBase Node, TensorInfo Info)[] parents = new (NodeBase, TensorInfo)[node.Parents.Count];
                        foreach ((NodeBuilder parent, int i) in node.Parents.Select((p, i) => (p, i)))
                            if (!map.TryGetValue(parent, out parents[i]))
                                return;

                        // Calculate the output tensor size
                        TensorInfo output = node.NodeType == ComputationGraphNodeType.Sum 
                            ? parents[0].Info 
                            : TensorInfo.Volume(parents[0].Info.Height, parents[0].Info.Width, parents.Sum(p => p.Info.Channels));
                        map[node] = (new MergeNode(node.NodeType, parents.Select(t => t.Node).ToArray()), output);
                        break;
                    default:
                        throw new ArgumentException($"Invalid node type: {node.NodeType}", nameof(root));
                }
                foreach (NodeBuilder child in node.Children)
                    BuildMap(child);
            }

            // Build the graph and bind the child nodes
            BuildMap(root);
            foreach (KeyValuePair<NodeBuilder, (NodeBase Node, TensorInfo Info)> pair in map)
            {
                pair.Value.Node.Children = pair.Key.Children.Select(child => map[child].Node).ToArray();
            }
            return map[root].Node;
        }

        /// <summary>
        /// Extracts the info on the input computation graph, and validates its structure
        /// </summary>
        /// <param name="root">The root <see cref="InputNode"/> for the computation graph</param>
        [Pure]
        private static (IReadOnlyList<IComputationGraphNode> Nodes, ProcessingNode output, IReadOnlyCollection<ProcessingNode> trainingOutputs) ExtractGraphInfo(InputNode root)
        {
            // Exploration setup
            if (root.Children.Any(child => !(child is ProcessingNode))) throw new ArgumentException("The nodes right after the graph root must be processing nodes");
            HashSet<IComputationGraphNode> nodes = new HashSet<IComputationGraphNode>();
            Dictionary<Guid, ProcessingNode> trainingOutputs = new Dictionary<Guid, ProcessingNode>();
            ProcessingNode output = null;

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
                                trainingOutputs.Add(trainingId, processing);
                            }
                            else if (output == null) output = processing;
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
                    case TrainingNode split:
                        if (trainingId != default) throw new ArgumentException("A training branch can't contain training split nodes");
                        for (int i = 0; i < split.Children.Count; i++)
                            if (!Explore(split.Children[i], Guid.NewGuid()))
                                return false;
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
